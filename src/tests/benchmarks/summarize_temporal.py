"""Validate paired benchmark inputs and write a compact comparison report."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    data = {}
    for name in ('orb', 'lightglue'):
        data[name] = {kind: json.loads((root/name/f'{kind}.json').read_text())
                      for kind in ('manifest', 'results', 'summary')}
    a, b = data['orb'], data['lightglue']
    assert [i['sha256'] for i in a['manifest']['items']] == [i['sha256'] for i in b['manifest']['items']]
    assert len(a['results']) == len(b['results']) == 400
    for x, y in zip(a['results'], b['results']):
        for key in ('index', 'source', 'mode', 'gps', 'transform', 'parameters', 'candidate_count'):
            assert x[key] == y[key], (key, x['index'])
        assert x['expected_candidate'] == (x['mode'] != 'wrong_place')
        assert y['expected_candidate'] == (y['mode'] != 'wrong_place')
    transformed = {name: sum(r['outcome']=='correct_id' for r in data[name]['results']
                              if r['mode'] in ('view','view_lighting') and r['original_mask_pixels'] > 0)
                   for name in ('orb','lightglue')}
    time_ratio = b['summary']['query_seconds']/a['summary']['query_seconds']
    verdict = (f"The coverage fix recovers {transformed['orb']}/190 transformed defect IDs with ORB, "
               f"versus 0 in the original run. SuperPoint + LightGlue recovers {transformed['lightglue']}/190 "
               f"using the same rule, with {time_ratio:.1f}× the total CPU query time in this implementation. "
               "These are synthetic repeat views, not evidence of seasonal reliability.")
    lines = ['# Temporal coverage fix and learned-feature evaluation', '',
        verdict, '', '100 test images, 400 queries per matcher, identical synthetic GPS and transformations.', '',
        '## Defect identity results', '',
        '| Scenario | ORB correct IDs | LightGlue correct IDs | ORB unresolved | LightGlue unresolved |',
        '|---|---:|---:|---:|---:|']
    for mode in ('identity', 'view', 'view_lighting', 'wrong_place'):
        counts = []
        for name in ('orb', 'lightglue'):
            rows = [r for r in data[name]['results'] if r['mode']==mode and r['original_mask_pixels'] > 0]
            from collections import Counter
            counts.append(Counter(r['outcome'] for r in rows))
        lines.append(f"| {mode} | {counts[0].get('correct_id',0)} | {counts[1].get('correct_id',0)} | {counts[0].get('unresolved',0)} | {counts[1].get('unresolved',0)} |")
    lines += ['', 'Each row reports the 95 images with nonempty ground-truth masks. Five empty-mask images are excluded from identity rates, but included in alignment and timing. Raw JSON retains all 100 queries per scenario. Wrong-place controls should remain unresolved.', '',
              '| Measure | ORB | SuperPoint + LightGlue |', '|---|---:|---:|']
    values = {}
    for name in ('orb','lightglue'):
        rows = data[name]['results']
        durations = sorted(r['elapsed_seconds'] for r in rows)
        values[name] = [
            data[name]['summary']['baseline_available'],
            sum(r['outcome']=='wrong_id' for r in rows),
            sum(r['outcome']=='duplicate_id' for r in rows),
            f"{statistics.median(durations):.3f} s",
            f"{durations[int(.95*(len(durations)-1))]:.3f} s",
            f"{sum(durations):.1f} s",
            f"{data[name]['summary']['reference_feature_bytes']/1024**2:.2f} MiB",
        ]
    labels = ['References with baseline IDs / 100', 'Wrong existing IDs / 400', 'New duplicate IDs / 400',
              'Median query time', '95th-percentile query time', 'Total query time', 'Compressed reference features']
    lines += [f'| {label} | {values["orb"][i]} | {values["lightglue"][i]} |' for i,label in enumerate(labels)]
    lines += ['', 'Query timing includes feature extraction, GPS candidate retrieval, matching against all candidates,',
              'coverage/overlap checks, and persistence. It excludes model loading, separate diagnostic alignment,',
              'and image export. CPU only; OpenCV uses one thread and PyTorch uses two. These are practical',
              'single-run measurements, not isolated hardware microbenchmarks.', '',
              '## Frame alignment, before the damage decision', '',
              '| Scenario | ORB aligned / 100 | LightGlue aligned / 100 |', '|---|---:|---:|---:|']
    for mode in ('identity','view','view_lighting'):
        lines.append(f"| {mode} | {a['summary']['alignment'][mode]} | {b['summary']['alignment'][mode]} |")
    lines += ['', '| Scenario | ORB median supported mask fraction | LightGlue median supported mask fraction |',
              '|---|---:|---:|']
    for mode in ('identity','view','view_lighting'):
        medians = []
        for name in ('orb','lightglue'):
            fractions = [r['mask_inside_source_hull'] for r in data[name]['results']
                         if r['mode']==mode and r.get('alignment')]
            medians.append(f'{100*statistics.median(fractions):.1f}%')
        lines.append(f'| {mode} | {medians[0]} | {medians[1]} |')
    lines += ['', 'Supported fractions are diagnostic source-hull coverage among successfully aligned true pairs.',
              'A wider feature hull can improve identity recovery even without improving descriptor robustness.',
              'This synthetic experiment does not establish cross-season performance.']
    lines += ['', '## How the coverage fix works', '',
        '1. GPS retrieves nearby reference frames. Feature matching and RANSAC estimate the road-plane homography.',
        '2. The source and reference inlier hulls are intersected in reference coordinates. This defines shared geometric support.',
        '3. At least 25% of each original damage mask and 64 foreground pixels must lie in that support. A small sliver cannot establish identity.',
        '4. Both masks and skeletons are clipped to shared support before the existing tolerant overlap comparison: three pixels, 60% match threshold, 25–60% ambiguous.',
        '5. One unambiguous matching ID is reused. Competing IDs remain unresolved. New IDs still require full support for the current mask.', '',
        'The old rule required the entire current mask to be supported before any comparison. The original benchmark recovered 1/100 unchanged IDs and 0/100 in each transformed group.',
        'The revised code permits partial evidence for identity without treating unseen pixels as absent damage.', '',
        '## What the learned matcher changes', '',
        'The optional adapter uses pretrained SuperPoint to extract up to 1,500 points and float descriptors, then LightGlue to pair them.',
        'ORB also uses a 1,500-feature cap. Both feed the same RANSAC, coverage, overlap, ambiguity, and storage code.',
        'Features are extracted at original resolution; learned points are filtered to the road ROI. Image dimensions are saved with features.',
        'Separate freshly built databases prevent mixing incompatible descriptors. SuperPoint + LightGlue is now the production default; ORB remains a comparison option.', '',
        '## Scope and limitations', '',
        '- The same 100 images previously used to diagnose the coverage failure are reused here. This is a paired development evaluation, not a fresh independent holdout.',
        '- The inherited harness creates an empty observation for five blank masks. Those cases are excluded from identity results; their empty reference masks contribute zero overlap to other queries. Raw enrollment counts include these artificial empty observations. The harness now emits no damage for empty masks; benchmark_used.py preserves the exact evaluated harness, and a regression test protects the correction.\n- Ground-truth masks are transformed with the photos; one complete binary annotation is one observation, which can include several crack branches. No segmentation inference is evaluated.',
        '- Random planar viewpoint changes: roll ±5°, pitch/yaw ±2°, scale 0.95–1.05, shifts ±2.5%. Lighting cases add gain 0.85–1.15, offset ±10, and mild Gaussian blur.',
        '- Synthetic GPS is spaced 20 m apart with reported 5 m accuracy and 2 m query jitter. Each query sees 4–7 candidate references. Real nearby imagery may be more confusing.',
        '- References are enrolled independently and then pooled. Queries never become references for subsequent queries.',
        '- The hull is geometric support, not a visibility detector. Vehicles, snow, shadows, and genuine seasonal changes are not handled or validated by this fix.',
        '- Crack-growth identity remains limited by symmetric overlap. No growth or repair inference was added.',
        '- The quarter-mask/64-pixel thresholds are heuristics fixed before comparing matchers; pixel thresholds depend on resolution.',
        '- Zero observed wrong matches is not proof that false associations cannot occur.', '',
        '## Reproduction', '', 'Run from the engine directory, choosing fresh output folders:', '', '```sh',
        'uv sync',
        'python -m src.tests.benchmarks.temporal_dataset --count 100 --seed 20260910 --matcher orb --output reports/comparison-rerun/orb',
        'python -m src.tests.benchmarks.temporal_dataset --count 100 --seed 20260910 --matcher lightglue --device cpu --output reports/comparison-rerun/lightglue',
        'python -m src.tests.benchmarks.summarize_temporal reports/comparison-rerun', '```', '',
        'The learned model package is pinned to official LightGlue commit eb42fee2d71449efb0aa5c10549752b5d75384d8.',
        'Each matcher directory contains manifest.json, results.json, summary.json, reference.sqlite, cases.jsonl, and transformed examples.',
        'The paired-input checks verified all 400 source/transform/GPS/candidate-count records agree between matchers.', '',
        'Implementation source: https://github.com/cvg/LightGlue', '']
    (root/'REPORT.md').write_text('\n'.join(lines))
    print(root/'REPORT.md')


if __name__ == '__main__':
    main()
