from __future__ import annotations

import json
from pathlib import Path
import re
import threading
from time import perf_counter
from typing import Any, Dict, Generator, Optional, List

from .llm_client import LLMClient
from .logger import logger
from .memory_os_local import MemoryOSLocal
from .profile_batch_updater import ProfileBatchUpdater
from .profile_schema import create_empty_static_profile, normalize_bare_profile
from .profile_utils import state_axis, context_axis, migrate_profile, runtime_profile_from_bare
from .utils import load_json, save_json, parse_json
from .prompts.prompt_loader import (
    DIRECT_RESPONSE_SYSTEM_PROMPT,
    DIRECT_RESPONSE_USER_PROMPT_TEMPLATE,
    PROFILE_EVOLUTION_SYSTEM_PROMPT,
    PROFILE_EVOLUTION_USER_PROMPT_TEMPLATE,
    USER_PROFILE_ACTIVATION_SYSTEM_PROMPT,
    USER_PROFILE_ACTIVATION_USER_PROMPT_TEMPLATE,
    UNDERSTANDING_FEEDBACK_SYSTEM_PROMPT,
    UNDERSTANDING_FEEDBACK_USER_PROMPT_TEMPLATE,
    EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT,
    EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE,
    PERIODIC_REBUILD_SYSTEM_PROMPT,
    PERIODIC_REBUILD_USER_PROMPT_TEMPLATE,
)
from .epistemic_decay import EpistemicDecayTracker, EXPLORATION_MODES

DEFAULT_CONFIG_PATH = "config.ini"
DEFAULT_PERSONA_PATH = "agent_persona.json"
DEFAULT_USER_DIR = "user"
DEFAULT_USER_NAME = "default_user"

FALLBACK_RESPONSE = "我刚才卡了一下，你再说一遍？"
MID_TERM_SOURCE_MESSAGES = 14

# Valid modeling modes
MODELING_MODES = ("explicit", "self_model", "flat")
# Valid update modes
UPDATE_MODES = ("bayesian_online", "static", "periodic_rebuild")


class StateDrivenCompanionAgent:
    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        profile_path: Optional[str] = None,
        persona_path: Optional[str] = None,
        user_name: str = DEFAULT_USER_NAME,
        # --- New experimental mode switches ---
        modeling_mode: str = "explicit",
        update_mode: str = "bayesian_online",
        exploration_mode: str = "adaptive",
        periodic_rebuild_interval: int = 5,
    ):
        # Validate modes
        if modeling_mode not in MODELING_MODES:
            raise ValueError(f"Unknown modeling_mode: {modeling_mode}. Must be one of {MODELING_MODES}")
        if update_mode not in UPDATE_MODES:
            raise ValueError(f"Unknown update_mode: {update_mode}. Must be one of {UPDATE_MODES}")
        if exploration_mode not in EXPLORATION_MODES:
            raise ValueError(f"Unknown exploration_mode: {exploration_mode}. Must be one of {EXPLORATION_MODES}")

        self.llm = LLMClient(config_path)
        self.user_name = user_name
        self.modeling_mode = modeling_mode
        self.update_mode = update_mode
        self.exploration_mode = exploration_mode
        self.periodic_rebuild_interval = periodic_rebuild_interval
        self._sessions_since_last_rebuild = 0

        self.profile_path = profile_path or self._profile_path_for_user(user_name)
        self.user_profile = self._load_or_create_user_profile(self.profile_path)
        self.persona_path = persona_path or DEFAULT_PERSONA_PATH
        # The persisted profile is the bare five-layer document. state_axis()
        # provides a runtime compatibility view for existing readers; do not
        # write legacy wrappers back into the profile file here.
        self.epistemic_tracker = EpistemicDecayTracker(mode=exploration_mode)
        self.last_empathy_state: Dict[str, Any] = {}
        self.last_prediction: Dict[str, Any] = {}
        self.last_agent_response: str = ""
        self.persona_config = load_json(self.persona_path)
        self.profile_batch_updater: Optional[ProfileBatchUpdater] = None
        if self.update_mode == "bayesian_online":
            self.profile_batch_updater = ProfileBatchUpdater(
                self.profile_path,
                on_profile_updated=self._on_profile_updated,
            )

        self.memory_manager = MemoryOSLocal()
        self._background_memory_running = False
        self._background_memory_lock = threading.Lock()
        self._background_generation = 0

    def _on_profile_updated(self, profile: Dict[str, Any]) -> None:
        # Keep runtime-only current/context state intact while refreshing the
        # persisted static profile produced by ProfileBatchUpdater.
        state_axis(self.user_profile)["static_profile"] = normalize_bare_profile(profile)

    def _save_profile(self) -> None:
        """Persist only the public bare five-layer Profile contract."""
        save_json(self.profile_path, normalize_bare_profile(state_axis(self.user_profile).get("static_profile", {})))

    def _profile_path_for_user(self, user_name: str) -> str:
        name = re.sub(r"[^0-9A-Za-z_\-一-鿿]+", "_", user_name).strip("_")
        return str(Path(DEFAULT_USER_DIR) / f"{name}_profile.json")

    def _load_or_create_user_profile(self, profile_path: str) -> Dict[str, Any]:
        path = Path(profile_path)

        if path.exists():
            stored = load_json(str(path))
            # Migrate legacy wrapper/leaf documents once, then keep the main
            # profile file in the fixed bare five-layer contract.
            bare = migrate_profile(stored)
            if stored != bare:
                save_json(str(path), bare)
            return runtime_profile_from_bare(bare)

        bare = create_empty_static_profile()
        save_json(str(path), bare)
        return runtime_profile_from_bare(bare)
    # ---------- prompt builders ----------
    def _prompt_context(
        self,
        user_input: str,
        relevant_memory: Dict[str, Any],
    ) -> Dict[str, str]:
        from .profile_utils import flatten_static_profile
        state = state_axis(self.user_profile)
        context = context_axis(self.user_profile)
        # Use flattened profile (key: value format, no nested JSON)
        flat_profile = flatten_static_profile(state.get("static_profile", {}))
        profile_lines = []
        for layer, attrs in flat_profile.items():
            if isinstance(attrs, dict):
                for k, v in attrs.items():
                    if v:
                        profile_lines.append(f"- {k}: {v}")
        profile_text = "\n".join(profile_lines) if profile_lines else json.dumps(flat_profile, ensure_ascii=False)
        return {
            "user_input": user_input,
            "static_profile": profile_text,
            "current_state": json.dumps(state.get("current_state", {}), ensure_ascii=False),
            "current_context": json.dumps(context, ensure_ascii=False),
            "persona_config": json.dumps(self.persona_config, ensure_ascii=False),
            "relevant_memory": json.dumps(relevant_memory, ensure_ascii=False),
        }

    def _response_prompt(
        self,
        user_input: str,
        relevant_memory: Dict[str, Any],
    ) -> str:
        return DIRECT_RESPONSE_USER_PROMPT_TEMPLATE.format(**self._prompt_context(user_input, relevant_memory))

    # ---------- memory background pipelines ----------
    def _start_background(self, target, args: tuple) -> None:
        with self._background_memory_lock:
            if self._background_memory_running:
                return
            self._background_memory_running = True
            generation = self._background_generation
        threading.Thread(target=target, args=args + (generation,), daemon=True).start()

    def _is_generation_stale(self, generation: int) -> bool:
        return generation != self._background_generation

    def _memory_pipeline(
        self,
        generation: int,
    ) -> None:
        try:
            threading.Event().wait(3)
            if self._is_generation_stale(generation):
                return
            self._run_memory_steps(generation)
        except Exception as e:
            logger.exception(f"[MEMORY_WRITE_ERROR] generation={generation} error={e}")
        finally:
            with self._background_memory_lock:
                if not self._is_generation_stale(generation):
                    self._background_memory_running = False

    def _run_memory_steps(self, generation: Optional[int] = None) -> None:
        def _step(name: str, fn):
            if self._is_generation_stale(generation):
                return
            start = perf_counter()
            try:
                result = fn()
                return result
            except Exception as e:
                logger.exception(f"[MEMORY_WRITE_STEP_ERROR] name={name} elapsed={perf_counter() - start:.3f}s error={e}")
                raise

        if len(self.memory_manager.short_term_memory) >= 20:
            _step("build_mid_term_summary",
                  lambda: self.memory_manager.build_mid_term_summary(self.llm, MID_TERM_SOURCE_MESSAGES))

        long_term_memory_id = _step(
            "extract_long_term_memory",
            lambda: self.memory_manager.extract_long_term_memory(self.llm),
        )
        if long_term_memory_id and self.profile_batch_updater is None:
            profile_updated = _step(
                "evolve_profile",
                lambda: self._evolve_profile_from_long_term(long_term_memory_id),
            )
            if profile_updated:
                self._save_profile()

    def _evolve_profile_from_long_term(self, long_term_memory_id: str) -> bool:
        """Evolve profile based on update_mode.

        - bayesian_online: incremental Bayesian update (our method)
        - static: no update at all
        - periodic_rebuild: handled separately in finalize_session
        """
        if self.update_mode == "static":
            return False

        if self.update_mode == "periodic_rebuild":
            return False

        # Default: bayesian_online
        long_term_memories = self.memory_manager.get_memories_by_ids([long_term_memory_id])
        if not long_term_memories:
            return False

        state = state_axis(self.user_profile)
        try:
            result = parse_json(self.llm.chat(
                PROFILE_EVOLUTION_SYSTEM_PROMPT,
                PROFILE_EVOLUTION_USER_PROMPT_TEMPLATE.format(
                    static_profile=json.dumps(state.get("static_profile", {}), ensure_ascii=False, indent=2),
                    long_term_memories=json.dumps(long_term_memories, ensure_ascii=False, indent=2),
                ),
                temperature=0.3,
            ))
            # Bayesian update output: {"reasoning": {...}, "static_profile": {...}}
            if isinstance(result, dict):
                updated_profile = result.get("static_profile", result)
                if isinstance(updated_profile, dict):
                    state["static_profile"] = updated_profile
                    return True
        except Exception as e:
            logger.exception(f"[MEMORY_PROFILE_EVOLVE_ERROR] error={e}")
            return False

    def rebuild_profile_from_conversations(self, conversations: List[Dict[str, Any]]) -> bool:
        """Rebuild profile from scratch using all conversation data (for periodic_rebuild mode).

        Args:
            conversations: List of conversation turn dicts.

        Returns:
            True if profile was successfully rebuilt.
        """
        from .profile_utils import flatten_static_profile
        state = state_axis(self.user_profile)
        conversation_text = "\n".join(
            f"{t.get('speaker', 'unknown')}: {t.get('content', '')}"
            for t in conversations
        )

        try:
            result = parse_json(self.llm.chat(
                PERIODIC_REBUILD_SYSTEM_PROMPT,
                PERIODIC_REBUILD_USER_PROMPT_TEMPLATE.format(
                    user_name=self.user_name,
                    full_conversation=conversation_text[:8000],
                ),
                temperature=0.3,
            ))
            if isinstance(result, dict):
                state["static_profile"] = result
                print(f"[Profile Rebuild] Complete rebuild from {len(conversations)} turns")
                return True
        except Exception as e:
            print(f"[Profile Rebuild Error] {e}")
            return False

    def _run_empathy_alignment(
        self,
        user_input: str,
        relevant_memory: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run Deep Empathy alignment reasoning with omega(t) modulation."""
        from .profile_utils import flatten_static_profile

        state = state_axis(self.user_profile)
        static_profile = state.get("static_profile", {})
        flattened = flatten_static_profile(static_profile)
        omega = self.epistemic_tracker.compute(static_profile)

        user_prompt = EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE.format(
            recent_context=json.dumps(relevant_memory, ensure_ascii=False)[:2000],
            user_message=user_input,
            user_profile=json.dumps(flattened, ensure_ascii=False)[:2000],
            agent_persona=json.dumps(self.persona_config, ensure_ascii=False)[:1000],
            current_state=json.dumps(state.get("current_state", {}), ensure_ascii=False),
            epistemic_omega=omega,
        )

        try:
            result = parse_json(self.llm.chat(
                EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT,
                user_prompt,
                temperature=0.3,
            ))
            if isinstance(result, dict):
                self.last_empathy_state = result.get("empathy_state", {})
                self.last_prediction = result.get("prediction", {})
            return result
        except Exception as e:
            print(f"[Empathy Alignment Error] {e}")
            return {}

    def _build_profile_activation_event(
        self,
        user_input: str,
        assistant_response: str,
        relevant_memory: Dict[str, Any],
        activation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(activation_result, dict):
            activation_result = {}

        activated_profile = activation_result.get("activated_profile", {})
        if not isinstance(activated_profile, dict):
            activated_profile = {}

        return {
            "type": "profile_activation",
            "user_message": user_input,
            "assistant_response": assistant_response,
            "activated_profile": activated_profile,
        }

    def _is_usable_profile_value(self, value: Any) -> bool:
        if value is None:
            return False
        text = str(value).strip()
        if not text:
            return False
        low_information_markers = (
            "未明确提及",
            "未提及",
            "没有明确提及",
            "不明确",
            "未知",
            "不详",
            "none",
            "null",
            "n/a",
        )
        return not any(marker in text.lower() for marker in low_information_markers)

    def _activation_profile_candidates(
        self,
        static_profile: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        candidates: Dict[str, Dict[str, Any]] = {}
        for layer, fields in static_profile.items():
            if not isinstance(fields, dict):
                continue
            layer_candidates: Dict[str, Any] = {}
            for field, payload in fields.items():
                value = payload.get("value") if isinstance(payload, dict) else payload
                if self._is_usable_profile_value(value):
                    layer_candidates[field] = value
            if layer_candidates:
                candidates[layer] = layer_candidates
        return candidates

    def _sanitize_profile_activation_result(
        self,
        activation_result: Dict[str, Any],
        candidates: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        layers = ("core", "regulation", "cognition", "identity", "behavior")
        raw_profile = activation_result.get("activated_profile", {})
        if not isinstance(raw_profile, dict):
            raw_profile = {}

        clean_profile = {layer: [] for layer in layers}

        for layer in layers:
            allowed_fields = candidates.get(layer, {})
            raw_items = raw_profile.get(layer, [])
            if isinstance(raw_items, dict):
                raw_items = [
                    {"field": field, **(item if isinstance(item, dict) else {"reason": str(item)})}
                    for field, item in raw_items.items()
                ]
            if not isinstance(raw_items, list):
                continue

            for item in raw_items:
                if isinstance(item, str):
                    field = item
                    reason = ""
                    confidence = 0.0
                elif isinstance(item, dict):
                    field = item.get("field")
                    reason = item.get("reason", "")
                    confidence = item.get("confidence", 0.0)
                else:
                    continue

                if field not in allowed_fields:
                    continue

                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = 0.0

                clean_profile[layer].append({
                    "field": field,
                    "value": allowed_fields[field],
                    "reason": reason,
                    "confidence": confidence,
                })

        return {
            "activated_profile": clean_profile,
        }

    def _run_user_profile_activation(
        self,
        user_input: str,
        assistant_response: str,
        relevant_memory: Dict[str, Any],
    ) -> Dict[str, Any]:
        state = state_axis(self.user_profile)
        static_profile = state.get("static_profile", {})
        candidates = self._activation_profile_candidates(static_profile)

        user_prompt = USER_PROFILE_ACTIVATION_USER_PROMPT_TEMPLATE.format(
            user_message=user_input,
            assistant_response=assistant_response,
            current_context=json.dumps(relevant_memory, ensure_ascii=False)[:2000],
            user_profile=json.dumps(candidates, ensure_ascii=False)[:4000],
        )

        try:
            result = parse_json(self.llm.chat(
                USER_PROFILE_ACTIVATION_SYSTEM_PROMPT,
                user_prompt,
                temperature=0.2,
                max_tokens=600,
            ))
            if isinstance(result, dict):
                return self._sanitize_profile_activation_result(result, candidates)
            return self._sanitize_profile_activation_result({}, candidates)
        except Exception as e:
            print(f"[User Profile Activation Error] {e}")
            return self._sanitize_profile_activation_result({}, candidates)

    def _run_profile_activation_log(
        self,
        user_input: str,
        assistant_response: str,
        relevant_memory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if relevant_memory is None:
            relevant_memory = self.memory_manager.retrieve_relevant_memory(user_input)
        activation_result = self._run_user_profile_activation(
            user_input,
            assistant_response,
            relevant_memory,
        )
        event = self._build_profile_activation_event(
            user_input=user_input,
            assistant_response=assistant_response,
            relevant_memory=relevant_memory,
            activation_result=activation_result,
        )
        logger.info("[PROFILE_ACTIVATION] " + json.dumps(event, ensure_ascii=False)[:2000])
        return event

    def _understanding_feedback(self, user_input: str) -> None:
        """UPDATING step: assess how previous empathy was received and update understanding."""
        if not self.last_agent_response:
            return

        from .profile_utils import flatten_static_profile
        state = state_axis(self.user_profile)
        static_profile = state.get("static_profile", {})
        flattened = flatten_static_profile(static_profile)

        try:
            feedback = parse_json(self.llm.chat(
                UNDERSTANDING_FEEDBACK_SYSTEM_PROMPT,
                UNDERSTANDING_FEEDBACK_USER_PROMPT_TEMPLATE.format(
                    previous_empathy_state=json.dumps(self.last_empathy_state, ensure_ascii=False),
                    previous_prediction=json.dumps(self.last_prediction, ensure_ascii=False),
                    agent_response=self.last_agent_response,
                    user_message=user_input,
                    user_profile=json.dumps(flattened, ensure_ascii=False)[:2000],
                ),
                temperature=0.3,
            ))
            if isinstance(feedback, dict):
                learning = feedback.get("learning", {})
                if learning.get("new_insight"):
                    print(f"[Understanding Update] {learning['new_insight']}")
                calibration = feedback.get("understanding_update", {})
                if calibration.get("calibration_note"):
                    print(f"[Calibration] {calibration['calibration_note']}")
        except Exception as e:
            print(f"[Understanding Feedback Error] {e}")


    def finalize_session(self) -> Dict[str, Any]:
        with self._background_memory_lock:
            self._background_generation += 1
            self._background_memory_running = False

        flushed_mid_term_ids = self.memory_manager.flush_short_term_memory(self.llm)
        long_term_memory_id = self.memory_manager.extract_long_term_memory(self.llm)
        if long_term_memory_id and self.profile_batch_updater is None:
            if self._evolve_profile_from_long_term(long_term_memory_id):
                self._save_profile()

        self._sessions_since_last_rebuild += 1
        if (self.update_mode == "periodic_rebuild"
                and self._sessions_since_last_rebuild >= self.periodic_rebuild_interval):
            all_messages = self.memory_manager.get_recent_messages(limit=100)
            self.rebuild_profile_from_conversations(all_messages)
            self._save_profile()
            self._sessions_since_last_rebuild = 0

        return {
            "flushed_mid_term_ids": flushed_mid_term_ids,
            "long_term_memory_id": long_term_memory_id
        }

    def observe_dialogue_turn(self, role: str, content: str) -> None:
        # UPDATING step runs in background so it never blocks the user's next interaction
        self.memory_manager.append_stm(role, content)
        self.epistemic_tracker.increment()
        if role == "user" and self.last_agent_response:
            threading.Thread(target=self._understanding_feedback, args=(content,), daemon=True).start()
        self._run_memory_steps()

    def chat_stream(
        self,
        user_input: str,
        ablate_dimension: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:

        self.memory_manager.append_stm("user", user_input)
        relevant_memory = self.memory_manager.retrieve_relevant_memory(user_input)

        # Empathy alignment reasoning runs in background; does NOT block streaming.
        # If a previous alignment result exists, use it; otherwise skip for this turn.
        empathy_state = self.last_empathy_state if self.last_empathy_state else {}
        threading.Thread(
            target=self._run_empathy_alignment,
            args=(user_input, relevant_memory),
            daemon=True,
        ).start()

        # Build the prompt
        response_prompt = self._response_prompt(user_input, relevant_memory)

        parts: List[str] = []
        first_token_logged = False
        t_start = perf_counter()
        try:
            for content in self.llm.chat_stream(
                DIRECT_RESPONSE_SYSTEM_PROMPT,
                response_prompt,
                temperature=0.4,
                max_tokens=450,
            ):
                if not first_token_logged:
                    print(f"[chat] first_token in {perf_counter() - t_start:.3f}s from interaction start")
                    first_token_logged = True
                parts.append(content)
                yield {"type": "token", "content": content}
        except Exception as e:
            print(f"[Stream Response Error] {e}")
            if not parts:
                parts.append(FALLBACK_RESPONSE)
                yield {"type": "token", "content": FALLBACK_RESPONSE}

        response = "".join(parts).strip() or FALLBACK_RESPONSE

        self.memory_manager.append_stm("assistant", response)
        self.last_agent_response = response
        self.epistemic_tracker.increment()
        if self.profile_batch_updater is not None:
            try:
                self.profile_batch_updater.submit_turn(user_input, response)
            except Exception as exc:
                logger.exception(f"[PROFILE_BATCH_ENQUEUE_ERROR] error={exc}")
        self._start_background(self._memory_pipeline, ())

        yield {
            "type": "done",
            "response": response,
            "background_memory_running": self._background_memory_running,
            "model_timing": self.llm.last_model_timing,
        }
