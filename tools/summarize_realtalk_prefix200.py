"""Offline speaker-paired uncertainty analysis; does not read or change prompts."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def analyze(payload, *, draws=10000, seed=20260909):
    per = payload['per_speaker']
    speakers = list(per)
    if len(speakers) != 10:
        raise ValueError('expected all ten speakers; no incomplete population inference')
    rng = random.Random(seed)
    resamples = [rng.choices(range(10), k=10) for _ in range(draws)]
    results = {}
    for metric in payload['macro']:
        direction = -1 if 'absolute_difference' in metric else 1
        deltas = [direction * (per[s][metric]['candidate'] - per[s][metric]['v9']) for s in speakers]
        boot = [statistics.mean(deltas[i] for i in sample) for sample in resamples]
        results[metric] = {
            'improvement_direction_mean': statistics.mean(deltas),
            'ci95_percentile': [percentile(boot, .025), percentile(boot, .975)],
            'improved_speakers': sum(d > 1e-12 for d in deltas),
            'worse_speakers': sum(d < -1e-12 for d in deltas),
            'tied_speakers': sum(abs(d) <= 1e-12 for d in deltas),
            'candidate_population_std': statistics.pstdev(per[s][metric]['candidate'] for s in speakers),
            'v9_population_std': statistics.pstdev(per[s][metric]['v9'] for s in speakers),
        }
    return {'seed': seed, 'draws': draws, 'unit': 'speaker paired bootstrap',
            'positive_means_improvement': True, 'metrics': results,
            'caveat': 'Exploratory early-prefix samples; intervals do not account for LLM or Judge rerun variability, prior tuning, or later-session sampling bias.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('comparison', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.comparison.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
