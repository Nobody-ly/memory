import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from summarize_realtalk_prefix200 import analyze, percentile


def test_paired_bootstrap_direction_and_constant_interval():
    fields = {'accuracy': {'v9': .4, 'candidate': .6},
              'intimacy_absolute_difference': {'v9': .08, 'candidate': .06}}
    result = analyze({'macro': fields, 'per_speaker': {str(i): fields for i in range(10)}}, draws=100)
    for field, delta in [('accuracy', .2), ('intimacy_absolute_difference', .02)]:
        assert result['metrics'][field]['ci95_percentile'] == pytest.approx([delta, delta])
        assert result['metrics'][field]['improved_speakers'] == 10
    assert percentile([0, 2], .5) == 1


def test_incomplete_speakers_rejected():
    with pytest.raises(ValueError):
        analyze({'macro': {}, 'per_speaker': {'one': {}}})
