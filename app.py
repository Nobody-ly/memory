import base64
import json
import os
import re

import atexit
from pathlib import Path

import copy
import threading
import time
from flask import Flask, Response, g, jsonify, request, stream_with_context
from flask_cors import CORS
from flask_sock import Sock
from dotenv import load_dotenv

load_dotenv()

from src.agent import StateDrivenCompanionAgent
from src.logger import logger
from src.profile_schema import create_empty_static_profile, normalize_bare_profile, validate_bare_profile
from src.utils import save_json, load_json

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)
sock = Sock(app)

agent = None
active_profile_id = None
active_persona_id = None
conversation_history = []
agent_state_condition = threading.Condition(threading.RLock())
agent_chat_lock = threading.Lock()
active_chat_count = 0

DATASET_OUTPUT_DIR = Path(os.getenv("MEMORY_DATASET_OUTPUT_DIR", "dataset/output_zh"))
DATASET_USER_DIR = DATASET_OUTPUT_DIR / "user"
DATASET_AGENT_DIR = DATASET_OUTPUT_DIR / "agent"
DATASET_TEST_USER_PROFILE = Path("dataset/new_user.json")
DATASET_TEST_AGENT_PERSONA = Path("dataset/test_agent.json")
WORKING_PROFILE_DIR = Path("data/active_profiles")
ALLOW_CREATE_USER_PROFILES = os.getenv(
    "MEMORY_ALLOW_CREATE_USER_PROFILES", "true"
).lower() in ("1", "true", "yes", "on")


def _humanize(name: str) -> str:
    """Convert a file-stem like 'fahim_khan' to a display name like 'Fahim Khan'."""
    return name.replace("_", " ").title()


def _validate_profile_id(profile_id: str) -> str:
    """Keep generated profile paths inside WORKING_PROFILE_DIR."""
    profile_id = (profile_id or "").strip()
    if not re.fullmatch(r"[0-9A-Za-z_\-]+", profile_id):
        raise ValueError("profile_id can only contain letters, numbers, '_' and '-'")
    return profile_id


def _working_profile_path(profile_id: str) -> Path:
    return WORKING_PROFILE_DIR / f"{profile_id}_profile.json"


def ensure_user_profile(profile_id: str) -> dict:
    profile_id = _validate_profile_id(profile_id)
    profile_info = USER_PROFILES.get(profile_id)
    if profile_info:
        source_path = Path(profile_info["source_path"])
        if source_path.exists():
            return profile_info
        logger.error(
            f"[PROFILE_INIT] registered profile source is missing: "
            f"profile_id={profile_id} path={source_path}"
        )

    if not ALLOW_CREATE_USER_PROFILES:
        raise ValueError(f"unknown user profile: {profile_id}")

    WORKING_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = _working_profile_path(profile_id)
    if not path.exists():
        save_json(str(path), create_empty_static_profile())
        logger.info(
            f"[PROFILE_INIT] created empty five-layer profile: "
            f"profile_id={profile_id} path={path}"
        )

    profile_info = {
        "id": profile_id,
        "display_name": _humanize(profile_id),
        "file_name": path.name,
        "source_path": str(path),
    }
    USER_PROFILES[profile_id] = profile_info
    return profile_info


def discover_user_profiles() -> dict:
    """Load the configured single test user profile."""
    profiles = {}
    if DATASET_TEST_USER_PROFILE.exists():
        profile_id = DATASET_TEST_USER_PROFILE.stem
        profiles[profile_id] = {
            "id": profile_id,
            "display_name": _humanize(profile_id),
            "file_name": DATASET_TEST_USER_PROFILE.name,
            "source_path": str(DATASET_TEST_USER_PROFILE),
        }
    if DATASET_USER_DIR.exists():
        for path in sorted(DATASET_USER_DIR.glob("*_profile.json")):
            profile_id = path.stem.removesuffix("_profile")
            profiles[profile_id] = {
                "id": profile_id,
                "display_name": _humanize(profile_id),
                "file_name": path.name,
                "source_path": str(path),
            }
    if WORKING_PROFILE_DIR.exists():
        for path in sorted(WORKING_PROFILE_DIR.glob("*_profile.json")):
            profile_id = path.stem.removesuffix("_profile")
            profiles.setdefault(profile_id, {
                "id": profile_id,
                "display_name": _humanize(profile_id),
                "file_name": path.name,
                "source_path": str(path),
            })
    return profiles


def discover_agent_personas() -> dict:
    """Load the single test agent persona from dataset/test_agent.json."""
    personas = {}
    if DATASET_TEST_AGENT_PERSONA.exists():
        personas["test_agent"] = {
            "id": "test_agent",
            "display_name": "Test Agent",
            "file_name": DATASET_TEST_AGENT_PERSONA.name,
            "source_path": str(DATASET_TEST_AGENT_PERSONA),
        }
    if DATASET_AGENT_DIR.exists():
        for path in sorted(DATASET_AGENT_DIR.glob("*_persona.json")):
            persona_id = path.stem.removesuffix("_persona")
            personas[persona_id] = {
                "id": persona_id,
                "display_name": _humanize(persona_id),
                "file_name": path.name,
                "source_path": str(path),
            }
    return personas

USER_PROFILES = discover_user_profiles()
AGENT_PERSONAS = discover_agent_personas()


def _wrap_dataset_profile(raw: dict) -> dict:
    """Normalize dataset or legacy input to the persisted bare five-layer profile."""
    return normalize_bare_profile(raw)

def finalize_agent_session() -> dict:
    if agent is None:
        return {
            "flushed_mid_term_ids": [],
            "long_term_memory_id": None,
            "remaining_short_term_count": 0,
        }
    return agent.finalize_session()

def finalize_agent_instance(agent_instance: StateDrivenCompanionAgent) -> None:
    try:
        updater = getattr(agent_instance, "profile_batch_updater", None)
        if updater is not None:
            updater.close()
        agent_instance.finalize_session()
    except Exception as e:
        logger.error(f"[FINALIZE_AGENT] error: {e}")

def require_agent():
    if agent is None:
        return jsonify({"error": "character is required"}), 409
    return None

def build_agent_for_character(profile_id: str, persona_id: str) -> StateDrivenCompanionAgent:
    profile_info = ensure_user_profile(profile_id)

    persona_info = AGENT_PERSONAS.get(persona_id)
    if not persona_info:
        raise ValueError(f"unknown agent persona: {persona_id}")

    source_profile_path = Path(profile_info["source_path"])
    persona_path = Path(persona_info["source_path"])
    if not source_profile_path.exists():
        raise FileNotFoundError(f"profile file not found: {source_profile_path}")
    if not persona_path.exists():
        raise FileNotFoundError(f"persona file not found: {persona_path}")

    # Load dataset profile and wrap into agent-expected format
    raw_profile = load_json(str(source_profile_path))
    wrapped = _wrap_dataset_profile(raw_profile)

    # Create the working copy once. Reuse it on later selections/restarts so
    # accumulated profile growth is not overwritten by the dataset seed.
    WORKING_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    working_profile_path = _working_profile_path(profile_id)
    if not working_profile_path.exists():
        save_json(str(working_profile_path), copy.deepcopy(wrapped))

    return StateDrivenCompanionAgent(
        profile_path=str(working_profile_path),
        persona_path=str(persona_path),
        user_name=profile_info["display_name"],
    )


atexit.register(finalize_agent_session)


@app.before_request
def _capture_system_start():
    val = request.headers.get("X-System-Start")
    try:
        g.system_start_ms = float(val) if val else None
    except (TypeError, ValueError):
        g.system_start_ms = None


def _cum_s():
    if g.system_start_ms:
        return (time.time() * 1000 - g.system_start_ms) / 1000
    return None


def _cum_str():
    c = _cum_s()
    return f" cumulative={c:.3f}s" if c is not None else ""


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/characters", methods=["GET"])
def get_characters():
    return jsonify({
        "user_profiles": [
            {"id": v["id"], "display_name": v["display_name"], "file_name": v["file_name"]}
            for v in USER_PROFILES.values()
        ],
        "agent_personas": [
            {"id": v["id"], "display_name": v["display_name"], "file_name": v["file_name"]}
            for v in AGENT_PERSONAS.values()
        ],
        "active_profile_id": active_profile_id,
        "active_persona_id": active_persona_id,
    }), 200


@app.route("/api/characters/select", methods=["POST"])
def select_character():
    global agent, active_profile_id, active_persona_id

    data = request.json or {}
    profile_id = data.get("profile_id") or data.get("character_id")
    persona_id = data.get("persona_id")

    if not profile_id:
        return jsonify({"error": "profile_id is required"}), 400
    if not persona_id:
        return jsonify({"error": "persona_id is required"}), 400

    with agent_state_condition:
        while active_chat_count:
            agent_state_condition.wait()

        if profile_id == active_profile_id and persona_id == active_persona_id and agent is not None:
            return jsonify({
                "message": "character already active",
                "profile_id": active_profile_id,
                "persona_id": active_persona_id,
                "profile": agent.user_profile,
                "profile_name": USER_PROFILES[profile_id]["display_name"],
                "persona_name": AGENT_PERSONAS[persona_id]["display_name"],
            }), 200

        previous_agent = agent
        agent = None
        active_profile_id = None
        active_persona_id = None
        if previous_agent is not None:
            finalize_agent_instance(previous_agent)

        try:
            next_agent = build_agent_for_character(profile_id, persona_id)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        agent = next_agent
        active_profile_id = profile_id
        active_persona_id = persona_id
        conversation_history.clear()
        return jsonify({
            "message": "character selected",
            "profile_id": active_profile_id,
            "persona_id": active_persona_id,
            "profile": agent.user_profile,
            "profile_name": USER_PROFILES[profile_id]["display_name"],
            "persona_name": AGENT_PERSONAS[persona_id]["display_name"],
        }), 200


@app.route("/api/chat", methods=["POST"])
def chat():
    global active_chat_count

    data = request.json or {}
    requested_profile_id = data.get("profile_id")
    requested_persona_id = data.get("persona_id")
    user_input = data.get("message", "").strip()
    ablate_dimension = data.get("ablate_dimension")

    if not requested_profile_id:
        return jsonify({"error": "profile_id is required"}), 400
    if not requested_persona_id:
        return jsonify({"error": "persona_id is required"}), 400
    if not user_input:
        return jsonify({"error": "message is required"}), 400

    with agent_state_condition:
        if agent is None:
            return jsonify({"error": "character is required"}), 409
        if (
            requested_profile_id != active_profile_id
            or requested_persona_id != active_persona_id
        ):
            return jsonify({
                "error": "character selection mismatch",
                "requested_profile_id": requested_profile_id,
                "active_profile_id": active_profile_id,
                "requested_persona_id": requested_persona_id,
                "active_persona_id": active_persona_id,
            }), 409
        local_agent = agent
        active_chat_count += 1

    t_start = time.perf_counter()

    def encode_event(payload):
        return json.dumps(payload, ensure_ascii=False) + "\n"

    @stream_with_context
    def generate():
        global active_chat_count
        first_token_time = None
        try:
            with agent_chat_lock:
                for event in local_agent.chat_stream(user_input, ablate_dimension=ablate_dimension):
                    event_type = event.get("type")

                    if event_type == "token":
                        if first_token_time is None:
                            first_token_time = time.perf_counter() - t_start
                        yield encode_event(event)
                        continue

                    if event_type == "profile_activation":
                        logger.info(
                            "[PROFILE_ACTIVATION] "
                            + json.dumps(event, ensure_ascii=False)[:2000]
                        )
                        yield encode_event(event)
                        continue

                    if event_type == "done":
                        response = event["response"]
                        total = time.perf_counter() - t_start
                        first_token = first_token_time if first_token_time is not None else total
                        logger.info(
                            f"[CHAT] first_token={first_token:.3f}s "
                            f"total={total:.3f}s chars={len(response)}"
                        )

                        conversation_history.append({"role": "user", "content": user_input})
                        conversation_history.append({"role": "assistant", "content": response})

                        yield encode_event({
                            "type": "done",
                            "message": response,
                            "profile": local_agent.user_profile,
                            "conversation_length": len(conversation_history),
                            "updated_fields": event.get("updated_fields", ["state_axis.current_state", "state_axis.projected_state"]),
                            "background_memory_running": event.get("background_memory_running", False),
                            "model_timing": event.get("model_timing"),
                            "usage": event.get("usage"),
                            "ablate_dimension": event.get("ablate_dimension"),
                            "activated_persona": event.get("activated_persona", {}),
                            "decision": event.get("decision", {}),
                        })
        except Exception as e:
            yield encode_event({"type": "error", "error": str(e)})
        finally:
            with agent_state_condition:
                active_chat_count -= 1
                agent_state_condition.notify_all()

    return Response(generate(), mimetype="application/x-ndjson")


@app.route("/api/profile-activation", methods=["POST"])
def profile_activation():
    agent_error = require_agent()
    if agent_error:
        return agent_error

    data = request.json or {}
    user_input = data.get("message", "").strip()
    assistant_response = data.get("assistant_response", "").strip()
    if not user_input:
        return jsonify({"error": "message is required"}), 400
    if not assistant_response:
        return jsonify({"error": "assistant_response is required"}), 400

    try:
        t_start = time.perf_counter()
        event = agent._run_profile_activation_log(
            user_input,
            assistant_response=assistant_response,
        )
        total = time.perf_counter() - t_start
        logger.info(f"[PROFILE_ACTIVATION_API] total={total:.3f}s")
        return jsonify(event or {"type": "profile_activation", "empty": True}), 200
    except Exception as e:
        logger.error(f"[PROFILE_ACTIVATION_API] error: {e}")
        return jsonify({"error": str(e)}), 500


def _handle_voice_msg(ws, data):
    msg_type = data.get("type")
    try:
        seq = int(data.get("seq", 0))
    except (TypeError, ValueError):
        seq = 0
    system_start_ms = data.get("system_start_ms")

    if msg_type == "asr":
        audio_b64 = data.get("audio", "")
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception:
            audio_bytes = b""
        from voice.asr import transcribe
        text = transcribe(audio_bytes, "audio.wav", system_start_ms=system_start_ms)
        try:
            ws.send(json.dumps({"type": "asr_result", "seq": seq, "text": text}))
        except Exception as e:
            logger.error(f"[WS] asr send failed: {e}")
    elif msg_type == "tts":
        text = data.get("text", "")
        from voice.tts import stream_tts_chunks
        try:
            stream_tts_chunks(text, ws, seq)
        except Exception as e:
            logger.error(f"[WS] tts failed seq={seq}: {e}")
        finally:
            try:
                ws.send(json.dumps({"type": "tts_end", "seq": seq}))
            except Exception as e:
                logger.error(f"[WS] tts_end send failed seq={seq}: {e}")


def _run_chat_via_ws(ws, message, system_start_ms, ablate_dimension):
    """Run one voice chat against an immutable snapshot of the active Agent."""
    global active_chat_count

    with agent_state_condition:
        if agent is None:
            try:
                ws.send(json.dumps({"type": "chat_error", "error": "agent not ready"}, ensure_ascii=False))
            except Exception:
                pass
            return
        local_agent = agent
        active_chat_count += 1

    t_start = time.perf_counter()
    first_token_time = None
    try:
        with agent_chat_lock:
            ws.send(json.dumps({"type": "chat_start", "system_start_ms": system_start_ms}, ensure_ascii=False))
            for event in local_agent.chat_stream(message, ablate_dimension=ablate_dimension):
                event_type = event.get("type")
                if event_type == "token":
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - t_start
                        logger.info(f"[CHAT] first_token={first_token_time:.3f}s input={message!r:.50}")
                    try:
                        ws.send(json.dumps({"type": "chat_token", "content": event.get("content", "")}, ensure_ascii=False))
                    except Exception as e:
                        logger.error(f"[WS] chat_token send failed: {e}")
                        return
                elif event_type == "done":
                    response = event["response"]
                    total = time.perf_counter() - t_start
                    logger.info(f"[CHAT] total={total:.3f}s chars={len(response)}")
                    ws.send(json.dumps({
                        "type": "chat_done",
                        "response": response,
                        "profile": local_agent.user_profile,
                        "updated_fields": event.get("updated_fields", []),
                        "background_memory_running": event.get("background_memory_running", False),
                    }, ensure_ascii=False))
    except Exception as e:
        logger.error(f"[WS] chat stream error: {e}")
        try:
            ws.send(json.dumps({"type": "chat_error", "error": str(e)}, ensure_ascii=False))
        except Exception:
            pass
    finally:
        with agent_state_condition:
            active_chat_count -= 1
            agent_state_condition.notify_all()


class _StreamASRSession:
    """Manages a dashscope streaming ASR session, forwarding events to the WS client."""

    def __init__(self, ws, sid, system_start_ms, ablate_dimension=None):
        import dashscope
        from dashscope.audio.asr import Recognition, RecognitionCallback
        from threading import Timer, Lock
        dashscope.api_key = os.getenv("API_KEY")
        self.ws = ws
        self.sid = sid
        self.system_start_ms = system_start_ms
        self.ablate_dimension = ablate_dimension
        self.accumulated = ""
        self.t_start = time.perf_counter()
        self.first_event_at = None
        self.last_event_at = None
        self._auto_end_timer = None
        self._end_lock = Lock()
        self._ended = False
        self.AUTO_END_DELAY_S = 1.5
        self.user_speech_start_ms = None
        self.user_speech_end_ms = None
        self.first_audio_received_ms = None  # wall clock when first audio chunk arrived
        self.acoustic_end_ms = None  # wall clock of actual speech end (via end_time field)

        session = self

        def _send(obj):
            try:
                session.ws.send(json.dumps(obj, ensure_ascii=False))
            except Exception as e:
                logger.error(f"[WS] asr stream send failed: {e}")

        def _schedule_auto_end():
            with session._end_lock:
                if session._ended:
                    return
                if session._auto_end_timer is not None:
                    session._auto_end_timer.cancel()
                session._auto_end_timer = Timer(session.AUTO_END_DELAY_S, session._do_auto_end)
                session._auto_end_timer.start()

        def _cancel_auto_end():
            with session._end_lock:
                if session._auto_end_timer is not None:
                    session._auto_end_timer.cancel()
                    session._auto_end_timer = None

        class CB(RecognitionCallback):
            def on_event(self, result):
                now = time.perf_counter()
                now_wall_ms = time.time() * 1000
                if session.first_event_at is None:
                    session.first_event_at = now
                session.last_event_at = now
                sentence = result.get_sentence() if hasattr(result, "get_sentence") else None
                if not sentence:
                    return
                if isinstance(sentence, list):
                    sentence = sentence[-1] if sentence else {}
                text = sentence.get("text", "") if isinstance(sentence, dict) else ""
                is_begin = sentence.get("sentence_begin", False) if isinstance(sentence, dict) else False
                is_end = sentence.get("sentence_end", False) if isinstance(sentence, dict) else False
                begin_time_ms = sentence.get("begin_time") if isinstance(sentence, dict) else None
                end_time_ms = sentence.get("end_time") if isinstance(sentence, dict) else None
                if is_begin:
                    if session.user_speech_start_ms is None:
                        session.user_speech_start_ms = now_wall_ms
                    if begin_time_ms is not None and session.first_audio_received_ms is None:
                        session.first_audio_received_ms = now_wall_ms - begin_time_ms
                if is_end:
                    session.user_speech_end_ms = now_wall_ms
                    if end_time_ms is not None and session.first_audio_received_ms is not None:
                        session.acoustic_end_ms = session.first_audio_received_ms + end_time_ms
                if is_end:
                    if text:
                        session.accumulated += text
                    _send({
                        "type": "asr_partial",
                        "session_id": session.sid,
                        "text": session.accumulated,
                    })
                    _schedule_auto_end()
                else:
                    _cancel_auto_end()
                    if text:
                        _send({
                            "type": "asr_partial",
                            "session_id": session.sid,
                            "text": session.accumulated + text,
                        })

            def on_complete(self):
                now_wall_ms = time.time() * 1000
                # Prefer acoustic_end (real mouth-close time); fall back to sentence_end arrival
                speech_end_value = session.acoustic_end_ms or session.user_speech_end_ms
                # acoustic_end is an estimate and can slightly exceed asr_complete; clamp it
                if speech_end_value and speech_end_value > now_wall_ms:
                    speech_end_value = now_wall_ms
                _send({
                    "type": "asr_final",
                    "session_id": session.sid,
                    "text": session.accumulated,
                    "speech_start_ms": int(session.user_speech_start_ms) if session.user_speech_start_ms else None,
                    "speech_end_ms": int(speech_end_value) if speech_end_value else None,
                    "asr_complete_ms": int(now_wall_ms),
                })
                # Only kick off chat if ASR produced non-empty text
                if not session.accumulated or not session.accumulated.strip():
                    _send({"type": "asr_empty", "session_id": session.sid})
                    return
                # Kick off chat immediately on the backend — no frontend round-trip
                threading.Thread(
                    target=_run_chat_via_ws,
                    args=(session.ws, session.accumulated, session.system_start_ms, session.ablate_dimension),
                    daemon=True,
                ).start()

            def on_error(self, result):
                logger.error(f"[ASR-STREAM] sid={session.sid} error: {result}")
                try:
                    session.ws.send(json.dumps({
                        "type": "asr_error",
                        "session_id": session.sid,
                        "error": str(result),
                    }))
                except Exception:
                    pass

        self.rec = Recognition(
            model="paraformer-realtime-v2",
            callback=CB(),
            format="pcm",
            sample_rate=16000,
            vad_silence_duration=400,
        )
        self.rec.start()
        logger.info(f"[ASR-STREAM] sid={sid} started")

    def feed(self, audio_chunk: bytes):
        try:
            self.rec.send_audio_frame(audio_chunk)
        except Exception as e:
            logger.error(f"[ASR-STREAM] feed error: {e}")

    def _do_auto_end(self):
        with self._end_lock:
            if self._ended:
                return
            self._ended = True
        logger.info(f"[ASR-STREAM-AUTO-END] sid={self.sid} triggered (no speech for {self.AUTO_END_DELAY_S}s after sentence_end)")
        try:
            self.rec.stop()
        except Exception as e:
            logger.error(f"[ASR-STREAM] auto-end stop error: {e}")

    def stop(self):
        with self._end_lock:
            if self._ended:
                return
            self._ended = True
            if self._auto_end_timer is not None:
                self._auto_end_timer.cancel()
                self._auto_end_timer = None
        try:
            self.rec.stop()
        except Exception as e:
            logger.error(f"[ASR-STREAM] stop error: {e}")


@sock.route("/ws/voice")
def voice_ws(ws):
    threads = []
    asr_session = {"session": None}
    session_lock = threading.Lock()

    def _cleanup_asr():
        with session_lock:
            sess = asr_session["session"]
            asr_session["session"] = None
        if sess:
            try:
                sess.stop()
            except Exception:
                pass

    while True:
        msg = ws.receive()
        if msg is None:
            break

        # Binary frame = raw PCM audio chunk
        if isinstance(msg, bytes):
            with session_lock:
                sess = asr_session["session"]
            if sess:
                sess.feed(msg)
            continue

        try:
            data = json.loads(msg)
        except (json.JSONDecodeError, TypeError):
            continue

        msg_type = data.get("type")

        if msg_type == "asr_stream_start":
            _cleanup_asr()
            sid = data.get("session_id") or f"asr_{int(time.time() * 1000)}"
            with session_lock:
                asr_session["session"] = _StreamASRSession(
                    ws=ws, sid=sid,
                    system_start_ms=data.get("system_start_ms"),
                    ablate_dimension=data.get("ablate_dimension"),
                )
            continue

        if msg_type == "asr_stream_end":
            _cleanup_asr()
            continue

        # TTS and legacy ASR run in worker threads
        t = threading.Thread(target=_handle_voice_msg, args=(ws, data), daemon=True)
        t.start()
        threads.append(t)

    _cleanup_asr()
    for t in threads:
        t.join(timeout=30)


@app.route("/api/log", methods=["POST"])
def client_log():
    data = request.json or {}
    logger.info(f"[CLIENT] {data.get('event','?')} {json.dumps({k:v for k,v in data.items() if k!='event'}, ensure_ascii=False)}")
    return jsonify({"ok": True})


@app.route("/api/profile", methods=["GET"])
def get_profile():
    agent_error = require_agent()
    if agent_error:
        return agent_error
    profile = normalize_bare_profile(agent.user_profile)
    return jsonify(profile), 200


@app.route("/api/profile", methods=["POST"])
def update_profile():
    agent_error = require_agent()
    if agent_error:
        return agent_error

    data = request.json or {}
    try:
        profile = validate_bare_profile(data)
        agent._on_profile_updated(profile)
        save_json(agent.profile_path, profile)
        return jsonify({"message": "profile updated", "profile": profile}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify({"history": conversation_history, "total": len(conversation_history)}), 200


@app.route("/api/reset", methods=["POST"])
def reset_chat():
    data = request.json or {}
    scope = data.get("scope", "chat")

    conversation_history.clear()

    if scope == "experiment":
        profile = agent.reset_to_initial_state()
        return jsonify({"message": "experiment reset", "profile": profile}), 200

    return jsonify({"message": "history reset"}), 200


@app.route("/api/finalize-session", methods=["POST"])
def finalize_session():
    try:
        return jsonify({"message": "session finalized", **finalize_agent_session()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=18201, use_reloader=False)
