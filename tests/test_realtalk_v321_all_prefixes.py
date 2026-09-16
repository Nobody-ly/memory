import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_realtalk_v321_all_prefixes import all_prefixes, validate_merged, v3


def test_all_prefixes_are_fixed_balanced_and_preserve_previous_sixty():
    prepared, _ = v3.base.prepare_formal_cb(Path(__file__).resolve().parents[1] / "dataset")
    windows = all_prefixes(prepared)
    assert len(windows) == 10
    assert sum(map(len, windows.values())) == 200
    for item in prepared:
        assert windows[item['speaker']] == [p['result_id'] for p in item['points'][:20]]
    from run_realtalk_v321_other_windows import choose_windows
    old, _ = choose_windows(prepared)
    assert [rid for s in ('Emi', 'Nicolas', 'Kevin') for rid in windows[s]] == old


def test_merge_rejects_missing_duplicate_and_unequal_people():
    rows = [{'result_id':f'{s}:{i}', 'speaker':str(s)} for s in range(10) for i in range(20)]
    expected = {r['result_id'] for r in rows}
    validate_merged(rows, expected)
    for bad in (rows[:-1], rows[:-1] + [rows[0]], [dict(r, speaker='0') for r in rows]):
        with pytest.raises(ValueError):
            validate_merged(bad, expected)
