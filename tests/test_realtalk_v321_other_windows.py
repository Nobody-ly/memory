import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_realtalk_v321_other_windows import choose_windows, summarize, v3


def test_windows_are_three_fixed_source_prefixes_without_scores():
    prepared, _ = v3.base.prepare_formal_cb(Path(__file__).resolve().parents[1] / "dataset")
    ids, windows = choose_windows(prepared)
    assert [w["speaker"] for w in windows] == ["Emi", "Nicolas", "Kevin"]
    assert len(ids) == len(set(ids)) == 60
    for i, speaker in enumerate(["Emi", "Nicolas", "Kevin"]):
        item = next(x for x in prepared if x["speaker"] == speaker)
        assert ids[i * 20:(i + 1) * 20] == [p["result_id"] for p in item["points"][:20]]
    assert all("Vanessa" not in rid for rid in ids)
    assert choose_windows(list(reversed(prepared)))[0] == ids


def test_macro_weights_speakers_equally_not_rows():
    rows = [{"speaker": "a", "metrics": {"score": 0.0}},
            {"speaker": "b", "metrics": {"score": 1.0}},
            {"speaker": "b", "metrics": {"score": 1.0}}]
    per, macro = summarize(rows, "metrics")
    assert macro["score"] == 0.5
    assert per["b"]["score"] == 1.0
