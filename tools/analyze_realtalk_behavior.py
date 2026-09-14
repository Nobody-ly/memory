"""Read-only, no-model diagnostics of matched REALTALK predictions and cached scores.

Text markers are proxies, never replacements for official metric labels. Cb strata
and diagnostic rules are retrospective, not held-out estimates of a new method.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.exp1_protocol import (  # noqa: E402
    REALTALK_PERSONA_SPLITS, build_message_level_points, build_profile_corpus,
    canonical_speaker,
)

PATTERNS = {
    "first_person": r"\b(?:i|i'm|i've|i'd|my|mine)\b",
    "reflection_marker": r"\b(?:i think|i feel|i felt|i guess|because|made me|helps me|i wonder|in my opinion)\b",
    "reason_marker": r"\b(?:because|since|that's why|that is why)\b",
    "affect_marker": r"\b(?:sad|happy|excited|worried|afraid|upset|love|hate|anxious|lonely|stressed|frustrated)\b",
    "negative_affect_marker": r"\b(?:sad|worried|afraid|upset|anxious|lonely|stressed|frustrated|hurt|depressed)\b",
    "advice_marker": r"\b(?:you should|you could|try to|recommend|suggest|make sure)\b",
    "support_marker": r"\b(?:sorry to hear|that sounds|must be|i understand|here for you|you deserve|proud of you)\b",
    "closure_marker": r"\b(?:bye|goodnight|good night|talk later|speak soon|see you)\b",
    "greeting_marker": r"\b(?:hi|hey|hello|good morning|good evening)\b",
    "opinion_request": r"\b(?:why|what do you think|how do you feel|your opinion|what makes you)\b",
}
LOCAL = ["rouge_l", "bertscore_f1", "sentiment_accuracy", "emotion_accuracy", "intimacy_absolute_difference"]
METRICS = LOCAL + ["reflectiveness_accuracy", "grounding_accuracy", "empathy_absolute_difference"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def avg(values):
    values = list(values)
    return statistics.mean(values) if values else None


def features(text):
    folded = text.casefold().replace("\u2019", "'")
    result = {k: bool(re.search(p, folded)) for k, p in PATTERNS.items()}
    result.update(characters=len(text), bubbles=len([s for s in text.splitlines() if s.strip()]),
                  question="?" in text or "\uff1f" in text,
                  question_marks=text.count("?") + text.count("\uff1f"))
    result["multi_bubble"] = result["bubbles"] > 1
    return result


def context_features(history, speaker, session):
    current = [t for t in history if t["session_id"] == session]
    # An old-session partner turn is not treated as a fresh request at a new opening.
    last = next((t for t in reversed(current) if t["speaker"].casefold() != speaker.casefold()), None)
    text = last["content"] if last else ""
    f = features(text)
    if not current:
        scene = "session_opening"
    elif last is None:
        scene = "no_current_partner_turn"
    elif f["question"]:
        scene = "explicit_question"
    elif f["closure_marker"]:
        scene = "closure_marker"
    elif f["negative_affect_marker"]:
        scene = "negative_affect_marker"
    elif f["affect_marker"]:
        scene = "other_affect_marker"
    else:
        scene = "other_statement"
    return scene, text, f, len(current)


def indexed(rows):
    result = {r["result_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate result_id")
    return result


def confusion(refs, preds):
    count = collections.Counter((bool(r), bool(p)) for r, p in zip(refs, preds, strict=True))
    tp, fp, fn, tn = (count[k] for k in [(True, True), (False, True), (True, False), (False, False)])
    return dict(n=tp+fp+fn+tn, tp=tp, fp=fp, fn=fn, tn=tn,
                accuracy=(tp+tn)/(tp+fp+fn+tn), reference_positive_rate=(tp+fn)/(tp+fp+fn+tn),
                predicted_positive_rate=(tp+fp)/(tp+fp+fn+tn),
                precision=tp/(tp+fp) if tp+fp else 0,
                recall=tp/(tp+fn) if tp+fn else 0,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)


def aggregate(rows):
    output = {"n": len(rows), "speakers": len(set(r["speaker"] for r in rows))}
    for variant in ["v9", "v2"]:
        output[variant] = {m: avg(r[variant]["metrics"][m] for r in rows) for m in METRICS}
        output[variant]["intimacy_signed_error"] = avg(r[variant]["intimacy_signed"] for r in rows)
        output[variant]["intimacy_over_rate"] = avg(r[variant]["intimacy_signed"] > 0 for r in rows)
        for m in ["reflectiveness", "grounding"]:
            output[variant][m] = confusion([r["reference_judge"][m] for r in rows], [r[variant]["judge"][m] for r in rows])
        output[variant]["text"] = {k: avg(r[variant]["text"][k] for r in rows) for k in rows[0][variant]["text"]}
    output["reference_text"] = {k: avg(r["reference_text"][k] for r in rows) for k in rows[0]["reference_text"]}
    return output


def strata(rows, key):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[str(r[key])].append(r)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


def behavior_profiles(turns, speaker):
    target = [t for t in turns if t["speaker"].casefold() == speaker.casefold()]
    fs = [features(t["content"]) for t in target]
    return {"n": len(target), **{k: avg(f[k] for f in fs) for k in features("")},
            "actual_source_bubbles_mean":avg(len(t['dia_ids']) for t in target),
            "actual_source_multibubble_rate":avg(len(t['dia_ids'])>1 for t in target)}


def strip_values(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return {k: strip_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strip_values(v) for v in value]
    return value


def causal_proxy_forecast(ca_turns, points, speaker):
    """Fixed-strength Ca prior plus past Cb observations; predicts markers, not Judge scores."""
    results = {}
    for marker in ["question", "reflection_marker", "multi_bubble"]:
        ca = [features(t["content"])[marker] for t in ca_turns if t["speaker"].casefold()==speaker.casefold()]
        prior = (sum(ca)+1)/(len(ca)+2)
        past=[]; observations=[]
        for point in points:
            probability=(8*prior+sum(past))/(8+len(past))
            actual=features(point['target_message'])[marker]
            observations.append({'ca_brier':(prior-actual)**2,'ca_cb_brier':(probability-actual)**2})
            past.append(actual)
        results[marker] = {key:avg(r[key] for r in observations) for key in observations[0]}
    return results


def analyze(inputs, dataset):
    manifest = read(inputs/'generation_manifest.json')
    for name,expected in manifest['dataset']['all_source_files_sha256'].items():
        if sha(dataset/name)!=expected:
            raise ValueError(f"Source file hash mismatch: {name}")
    candidates, baselines = indexed(lines(inputs/"candidate.jsonl")), indexed(lines(inputs/"baseline.jsonl"))
    cj, bj = indexed(lines(inputs/"candidate_judge.jsonl")), indexed(lines(inputs/"baseline_judge.jsonl"))
    if not (set(candidates) == set(baselines) == set(cj) == set(bj)) or len(candidates) != 519:
        raise ValueError("Require four identical complete 519-ID sets")
    raw = {r["response_id"]: r for r in lines(inputs/"raw.jsonl") if r.get("response_id")}
    source = {}
    profiles = {}
    forecasts = {}
    for split in REALTALK_PERSONA_SPLITS:
        ca, cb = read(dataset/split["train_chat"]), read(dataset/split["test_chat"])
        speaker = canonical_speaker(cb, split["speaker"])
        points = build_message_level_points(cb, speaker, test_sessions=3, max_context_chars=0, merge_adjacent_bubbles=True)
        ca_turns = build_profile_corpus(ca, speaker, 3)["turns"]
        cb_turns = [p["target"] for p in points]
        profiles[speaker] = {"ca": behavior_profiles(ca_turns, speaker), "cb_reference_retrospective": behavior_profiles(cb_turns, speaker)}
        forecasts[speaker] = causal_proxy_forecast(ca_turns, points, speaker)
        for p in points:
            source[(speaker.casefold(), p["target"]["turn_id"])] = p
    rows, audits = [], collections.Counter()
    for rid, candidate in candidates.items():
        baseline, judge, old_judge = baselines[rid], cj[rid], bj[rid]
        point = source[(candidate["speaker"].casefold(), candidate["target_turn_id"])]
        if not (candidate["ground_truth"] == baseline["ground_truth"] == point["target_message"]):
            raise ValueError(f"Ground truth mismatch: {rid}")
        if candidate["history_hash"] != point["history_hash"]:
            raise ValueError(f"Candidate causal history mismatch: {rid}")
        if baseline["context_hash"] != point["history_hash"]:
            raise ValueError(f"V9 causal history mismatch: {rid}")
        if judge["reference"] != old_judge["reference"]:
            raise ValueError(f"Unshared Judge reference: {rid}")
        if candidate["local_labels"]["reference"] != baseline["local_labels"]["reference"]:
            raise ValueError(f"Unshared local reference: {rid}")
        decision = candidate["decision"]
        response = raw.get(candidate["decision_audit"]["response_id"])
        if response is None:
            raise ValueError(f"Missing original decision response: {rid}")
        try:
            raw_decision = json.loads(response["raw_response"])
            audits["raw_decision_exact_match"] += raw_decision == decision
            audits["raw_decision_different"] += raw_decision != decision
            audits["raw_decision_equal_after_strip"] += strip_values(raw_decision) == strip_values(decision)
        except json.JSONDecodeError:
            audits["raw_decision_not_plain_json"] += 1
        scene, last, last_features, current_count = context_features(point["context_turns"], candidate["speaker"], point["target_session"])
        policy = decision["behavior_policy"]
        row = {"result_id": rid, "speaker": candidate["speaker"], "session": candidate["target_session"],
               "turn_index": point["message_level_index"], "scene": scene,
               "position": "opening" if current_count == 0 else "early" if current_count < 4 else "interior",
               "history_turns": len(point["context_turns"]), "last_partner": last,
               "partner_features": last_features, "reference": candidate["ground_truth"],
               "reference_text": features(candidate["ground_truth"]), "reference_judge": judge["reference"],
               "policy": policy, "alignment": decision["alignment"],
               "question_policy": policy["question_policy"], "reflection_policy": policy["reflection_policy"],
               "primary_goal": policy["primary_goal"], "user_need": decision["user_state"]["interaction_need"],
               "attempts": candidate["decision_audit"]["logical_attempts"]}
        for variant, pred, scored in [("v2",candidate,judge),("v9",baseline,old_judge)]:
            local = pred["local_labels"]
            row[variant] = {"message": pred["generated_message"], "text": features(pred["generated_message"]),
                            "judge": scored["candidate"], "metrics": {**pred["local_metrics"], **scored["metrics"]},
                            "intimacy_signed": local["candidate"]["intimacy"] - local["reference"]["intimacy"]}
        row["question_none_with_mark"] = policy["question_policy"] == "none" and row["v2"]["text"]["question"]
        row["reflection_none_judge_positive"] = policy["reflection_policy"] == "none" and judge["candidate"]["reflectiveness"]
        row["optional_disclosure_conflict_proxy"] = policy["self_disclosure_policy"] == "none" and any("self_disclos" in t.lower().replace("-","_").replace(" ","_") for t in policy["optional_content"])
        row["optional_question_conflict_proxy"] = policy["question_policy"] == "none" and any("question" in t.lower() for t in policy["optional_content"])
        row["reference_source_bubbles"] = len(point['target']['dia_ids'])
        row["reference_multibubble"] = row["reference_source_bubbles"]>1
        row["generated_support_marker"] = row["v2"]["text"]["support_marker"]
        rows.append(row)
    for speaker in profiles:
        group = sorted([r for r in rows if r["speaker"] == speaker], key=lambda r:r["turn_index"])
        for sess in {r["session"] for r in group}:
            for i, r in enumerate(r for r in group if r["session"] == sess):
                r["prefix_band"] = "first4_per_session" if i<4 else "remaining_per_session"
    summary = {"validation": {"records":len(rows), "all_gt_history_and_reference_labels_match":True,
                              "all_10_source_sha256_match":True,
                              "reference_source_bubbles":sum(r['reference_source_bubbles'] for r in rows),**audits},
               "micro":aggregate(rows), "speaker":strata(rows,"speaker"), "session":strata(rows,"session"),
               "scene":strata(rows,"scene"), "position":strata(rows,"position"), "prefix_band":strata(rows,"prefix_band"),
               "question_policy":strata(rows,"question_policy"), "reflection_policy":strata(rows,"reflection_policy"),
               "primary_goal":strata(rows,"primary_goal"), "user_need":strata(rows,"user_need"),
               "reference_multibubble":strata(rows,"reference_multibubble"),
               "generated_support_marker":strata(rows,"generated_support_marker"),
               "profiles":profiles,
               "contracts":{key:sum(r[key] for r in rows) for key in ["question_none_with_mark","reflection_none_judge_positive","optional_disclosure_conflict_proxy","optional_question_conflict_proxy"]}}
    summary["macro"] = {v:{m:avg(s[v][m] for s in summary["speaker"].values()) for m in METRICS} for v in ["v9","v2"]}
    summary["macro_signed_intimacy"] = {v:avg(s[v]["intimacy_signed_error"] for s in summary["speaker"].values()) for v in ["v9","v2"]}
    saved_report = read(inputs/'paired_report.json')
    for metric in METRICS:
        for variant,report_variant in [('v9','v9'),('v2','v15')]:
            if abs(summary['macro'][variant][metric]-saved_report['speaker_macro'][metric][report_variant])>0.000001:
                raise ValueError(f"Saved macro report mismatch: {metric}/{variant}")
    domains=read(inputs/'self_domains.json')
    conditions=[c for d in domains.values() for c in d['behavioral_conditions']]
    self_raw=[]
    self_parse_failures=0
    for response in raw.values():
        if response['operation_key'].startswith('self:'):
            try:
                self_raw.append(json.loads(response['raw_response']))
            except json.JSONDecodeError:
                self_parse_failures+=1
    summary['self_domain_audit']={
        'conditions':len(conditions),
        'trigger_counts':dict(collections.Counter(c['trigger'] for c in conditions)),
        'by_speaker':{s:dict(collections.Counter(c['trigger'] for c in d['behavioral_conditions'])) for s,d in domains.items()},
        'raw_self_invalid_json_attempts':self_parse_failures,
        'all_profiles_present_in_raw_after_strip':all(any(strip_values(r)==strip_values(d) for r in self_raw) for d in domains.values()),
        'examples':[{k:c[k] for k in ['trigger','likely_response','question_tendency','reflection_tendency']}
                    for c in domains['Emi']['behavioral_conditions']],
        'cause_not_established':'Raw model outputs show collapsed categories; schema compliance does not validate semantics.'}
    summary["ca_vs_causal_cb_proxy_forecast"] = {"prior_strength":8,"ca_beta_smoothing":[1,1],
        "settings_not_fitted_to_cb":True,"metric":"Brier score, lower is better; text markers only",
        "by_speaker":forecasts,
        "speaker_macro":{marker:{key:avg(f[marker][key] for f in forecasts.values()) for key in ['ca_brier','ca_cb_brier']}
                         for marker in ['question','reflection_marker','multi_bubble']}}
    duplicate_groups=collections.defaultdict(list)
    for r in rows:
        for variant in ['reference','v9','v2']:
            message=r['reference'] if variant=='reference' else r[variant]['message']
            labels=r['reference_judge'] if variant=='reference' else r[variant]['judge']
            duplicate_groups[message.strip()].append({'result_id':r['result_id'],'variant':variant,'labels':labels})
    summary['identical_text_label_disagreement_context_not_matched'] = [
        {'text':text,'occurrences':items,'does_not_establish_judge_inconsistency':True} for text,items in duplicate_groups.items()
        if len(items)>1 and len({json.dumps(i['labels'],sort_keys=True) for i in items})>1]
    summary["proxy_diagnostics"] = {}
    summary["paired_transitions"] = {}
    for metric, proxy in [("reflectiveness","reflection_marker"),("grounding","question")]:
        summary["proxy_diagnostics"][metric] = {}
        for name in ["reference","v9","v2"]:
            target = [r["reference_judge"][metric] if name=="reference" else r[name]["judge"][metric] for r in rows]
            pred = [r["reference_text"][proxy] if name=="reference" else r[name]["text"][proxy] for r in rows]
            summary["proxy_diagnostics"][metric][name] = confusion(target,pred)
        transitions = collections.Counter()
        for r in rows:
            old, new, ref = r["v9"]["judge"][metric],r["v2"]["judge"][metric],r["reference_judge"][metric]
            key = ("both_correct" if old==ref and new==ref else "both_wrong" if old!=ref and new!=ref
                   else "fixed_false_negative" if new==ref and ref else "fixed_false_positive" if new==ref
                   else "new_false_negative" if ref else "new_false_positive")
            transitions[key]+=1
        summary["paired_transitions"][metric] = dict(transitions)
    # This null baseline diagnoses label imbalance; it is not a proposed generation policy.
    summary["always_negative_diagnostic_only"] = {m:avg(avg(not r["reference_judge"][m] for r in rows if r["speaker"]==s) for s in profiles) for m in ["reflectiveness","grounding"]}
    summary["empathy_components"] = {}
    for v in ["v9","v2"]:
        summary["empathy_components"][v] = {component:{
            "mean_signed_error":avg(r[v]["judge"]["empathy"][component]-r["reference_judge"]["empathy"][component] for r in rows),
            "mean_absolute_error":avg(abs(r[v]["judge"]["empathy"][component]-r["reference_judge"]["empathy"][component]) for r in rows),
            "false_positive_zero_reference":sum(r["reference_judge"]["empathy"][component]==0 and r[v]["judge"]["empathy"][component]>0 for r in rows),
        } for component in ["emotional_reaction","interpretation","exploration"]}
    summary["lambda"] = {orientation:{"n":len(group),"mean":avg(r["alignment"]["lambda_trace"] for r in group),
                                      "min":min(r["alignment"]["lambda_trace"] for r in group),"max":max(r["alignment"]["lambda_trace"] for r in group)}
                         for orientation in sorted({r["alignment"]["orientation"] for r in rows})
                         if (group:=[r for r in rows if r["alignment"]["orientation"]==orientation])}
    # Contiguous ten-target windows never cross a speaker/session boundary.
    windows=[]
    for speaker in profiles:
        for session in ["session_1","session_2","session_3"]:
            group=sorted([r for r in rows if r["speaker"]==speaker and r["session"]==session],key=lambda r:r["turn_index"])
            for start in range(len(group)-9):
                block=group[start:start+10]; stats=aggregate(block)
                windows.append({"speaker":speaker,"session":session,"ids":[r["result_id"] for r in block],
                                "stats":stats,"combined_accuracy_delta":sum(stats['v2'][m]-stats['v9'][m] for m in ['reflectiveness_accuracy','grounding_accuracy'])})
    summary["worst_contiguous_windows_retrospective"] = sorted(windows,key=lambda w:w["combined_accuracy_delta"])[:10]
    summary["case_ids"] = {}
    for m in ["reflectiveness","grounding"]:
        for ref in [False,True]:
            key=f"{m}_new_{'fn' if ref else 'fp'}"
            eligible=[r for r in rows if r["reference_judge"][m]==ref and r["v9"]["judge"][m]==ref and r["v2"]["judge"][m]!=ref]
            seen=set(); selected=[]
            for r in sorted(eligible,key=lambda r:(r["turn_index"],r["speaker"])):
                if r["speaker"] not in seen:
                    selected.append(r["result_id"]); seen.add(r["speaker"])
                if len(selected)==5:break
            summary["case_ids"][key]=selected
    return rows,summary


def write_outputs(output,rows,summary,inputs,dataset):
    output.mkdir(parents=True,exist_ok=True)
    def dump(name,obj):
        (output/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
    dump("summary.json",summary)
    with (output/"samples.jsonl").open("w",encoding="utf-8") as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    flat=[]
    for r in rows:
        item={k:r[k] for k in ["result_id","speaker","session","scene","position","prefix_band","question_policy","reflection_policy","primary_goal","question_none_with_mark"]}
        for v in ["v9","v2"]:
            item.update({f"{v}_{k}":n for k,n in r[v]["metrics"].items()})
        flat.append(item)
    with (output/"samples.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    manifest={"analysis":"cached_scores_only_v1","model_calls":0,"inference_or_training":False,
              "reference_labels_used_for_retrospective_diagnosis_only":True,
              "proxy_rules_are_not_official_metrics":True,
              "script_sha256":sha(__file__),
              "inputs":{str(p.resolve()):sha(p) for p in list(inputs.iterdir())+list(dataset.glob('Chat_*.json')) if p.is_file()}}
    dump("manifest.json",manifest)
    text=["# REALTALK V2 offline diagnosis", "", "519 matched rows. Speaker macro is the main aggregation; strata below are message micro.",
          "No new model calls. Cb labels and windows are retrospective diagnostics, not new held-out results.", "",
          "| Metric | V9 | V2 |", "|---|---:|---:|"]
    for m in METRICS:text.append(f"| {m} | {summary['macro']['v9'][m]:.6f} | {summary['macro']['v2'][m]:.6f} |")
    for dimension in ["speaker","session","scene","prefix_band","question_policy","reflection_policy"]:
        text.extend(["",f"## {dimension}","", "| Group | N | Reflect V9 / V2 | Ground V9 / V2 | Intimacy AD V9 / V2 | Empathy AD V9 / V2 |", "|---|---:|---:|---:|---:|---:|"])
        for key,s in summary[dimension].items():
            values=[f"{s['v9'][m]:.3f} / {s['v2'][m]:.3f}" for m in ['reflectiveness_accuracy','grounding_accuracy','intimacy_absolute_difference','empathy_absolute_difference']]
            text.append(f"| {key} | {s['n']} | "+" | ".join(values)+" |")
    text.extend(["","## Audit","","```json",json.dumps({k:summary[k] for k in ['validation','contracts','paired_transitions','empathy_components','proxy_diagnostics','always_negative_diagnostic_only','lambda']},ensure_ascii=False,indent=2),"```"])
    (output/"diagnostics.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    cases=["# Retrospective case review", "", "Labels below are cached Judge outputs, not independently verified human labels."]
    index={r['result_id']:r for r in rows}
    for kind,ids in summary['case_ids'].items():
        cases.extend(["",f"## {kind}"])
        for rid in ids:
            r=index[rid]
            cases.extend(["",f"### {rid}","",f"Partner: {r['last_partner']}","",f"Reference: {r['reference']}","",f"V9: {r['v9']['message']}","",f"V2: {r['v2']['message']}","","Policy:","```json",json.dumps(r['policy'],ensure_ascii=False,indent=2),"```"])
    (output/"cases.md").write_text("\n".join(cases)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows,summary=analyze(args.inputs,args.dataset)
    write_outputs(args.output,rows,summary,args.inputs,args.dataset)
    print(json.dumps({"validation":summary['validation'],"macro":summary['macro'],"contracts":summary['contracts']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
