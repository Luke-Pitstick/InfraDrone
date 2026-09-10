# Temporal matching: 100 held-out road images

The current engine does not reliably recover defect IDs in this test. It rejects
nearly all repeat sightings, including unchanged images. The primary observed
failure is the requirement that the entire damage mask lie inside the matched
feature hull, rather than failure to align the pavement.

## Results

Each scenario contains 100 independent queries against the same reference catalog.
Queries are removed after evaluation so earlier query observations cannot affect
later outcomes.

| Scenario | Correct original ID | Unresolved | Wrong ID / duplicate ID | True-pair alignment |
|---|---:|---:|---:|---:|
| Identical image | 1 | 99 | 0 / 0 | 99 |
| Random viewpoint | 0 | 100 | 0 / 0 | 94 |
| Random viewpoint + lighting and mild blur | 0 | 100 | 0 / 0 | 89 |
| Wrong GPS neighborhood (negative control) | N/A | 100 | 0 / 0 | N/A |

99 of 100 references established a baseline ID; one lacked sufficient features.
The positive-query denominator remains 100, including that enrollment failure.
GPS retrieved the correct reference for every positive query, alongside unrelated
references (4–7 candidates total). Negative controls had 4–7 unrelated references
and excluded the true source; all abstained rather than assigning a wrong ID.
This zero observed false-match result is limited to this sample, not a guarantee.

## Diagnosis

- Of 99 aligned identical pairs, 98 had at least one foreground pixel outside the
  source feature hull. The remaining pair recovered its ID.
- All 94 aligned viewpoint pairs had foreground outside the hull. Median fraction
  inside was 68.2%; median of pairwise median transform errors was 0.111 pixels.
- All 89 aligned viewpoint-plus-lighting pairs had foreground outside the hull.
  Median fraction inside was 67.8%; median transform error was 0.120 pixels.

Transform errors compare the estimated homography with the known synthetic
homography at the matched inlier points. These are diagnostic fits, rerun using the
same helpers against the true reference; they are not stored production confidence
scores. The hull requirement is an exact all-pixels check. Cracks reaching the
image border cannot usually satisfy it because their surrounding ORB features are
in the interior. The prior random-texture tests used centrally contained masks and
did not reveal this behavior.

The next change to evaluate is matching only within reliably shared visible
coverage, with a minimum amount of supporting damage evidence. Simply deleting
the coverage check would allow poorly supported extrapolation. No production
thresholds or matching rules were changed during this benchmark.

## Setup and limits

- Seed: `20260910`; 100 images chosen uniformly without replacement from the 1,695
  JPEG images in `datasets/segmentation/crack_segmentation_dataset/test/images`.
  No images were rejected based on feature quality or matching outcome.
- Binary masks come from the corresponding test masks, thresholded at 128 because
  they are JPEG files. One full annotation mask is treated as one observation;
  it may contain multiple crack branches or components. This tests whole-mask
  matching and does not reproduce the segmentation engine's instance splitting.
- Full-image masks and skeletons transform with the image. The detector is not
  run, so results isolate temporal matching from segmentation error.
- Synthetic coordinates begin at 40° N, 105° W with 20-meter northward spacing,
  reported accuracy 5 m, and independent Gaussian query jitter of 2 m per axis.
  Coordinates are artificial; unrelated test images do not form a real road.
- References are enrolled independently using the actual engine, then their route
  IDs are pooled into one catalog. This controls the known existing catalog and
  exercises GPS distractors. It does not test sequential initial-map construction.
- Viewpoint homographies use a synthetic pinhole camera over a plane: roll ±5°,
  pitch/yaw perturbations ±2°, scale 0.95–1.05, lateral shifts ±2.5% of the image.
  Each non-identity scenario draws new independent transformations.
- Lighting cases additionally use gain 0.85–1.15, brightness offset ±10 intensity
  levels, and Gaussian blur sigma 0.6 with a 3×3 kernel. Exposed borders are black;
  no reflection padding invents pavement. All transformed views are retained,
  including clipping and weak-feature cases.
- Intrinsics and camera height are synthetic. This experiment does not validate
  metric dimensions, GPS georeferencing, actual seasonal changes, 3D potholes,
  rolling shutter, moving shadows, or independent future-survey imagery.

## Artifacts and reproduction

- `manifest.json`: selected source paths and hashes, synthetic GPS, calibration,
  reference IDs, runtime versions, and engine source hashes.
- `results.json`: all 400 outcomes, exact transformations, query GPS, timing,
  candidate counts, and registration diagnostics.
- `reference.sqlite`: the actual reference database, excluding query observations.
- `view/`, `view_lighting/`: 200 transformed PNG images and matching mask PNGs.
- `examples.jpg`: eight original/transformed pairs with red mask overlays.
- `summary.json`: aggregate counts. Initial benchmark duration: 37.3 seconds,
  excluding subsequent export of all transformed images.

From the engine directory, choose a new output folder:

```sh
.venv/bin/python -m src.tests.benchmarks.temporal_dataset --count 100 --seed 20260910 --output reports/temporal-100-rerun
```

The regular 33-test suite still passes. That does not override the poor repeat-ID
recovery observed by this larger dataset benchmark.
