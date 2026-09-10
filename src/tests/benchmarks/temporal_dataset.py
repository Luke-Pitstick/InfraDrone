"""Reproducible held-out-image benchmark, run from the engine root as a module.

Ground-truth masks isolate temporal matching from segmentation. Synthetic GPS and
camera intrinsics are test fixtures, not georeferencing claims about the dataset.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import platform

import cv2
import numpy as np
from skimage.morphology import skeletonize

from src.engine.calibration import CameraCalibration
from src.engine.constants import DamageType
from src.engine.models import Damage
from src.engine.motion import extract_features, match_features, fit_homography
from src.engine.temporal import Association, TemporalEngine
from src.engine.temporal_store import TemporalStore
from src.engine.video import Location, Video, VideoFrame
from src.engine.video_pipeline import FrameResult


def observation(image, mask, location):
    damage = Damage(DamageType.CRACK, 1.0, mask=mask,
                    skeleton=skeletonize(mask > 0).astype(np.uint8))
    return FrameResult(VideoFrame(image, 0, 0), location, [damage] if mask.any() else [])


def transform(rng, size, mode):
    if mode in ('identity', 'wrong_place'):
        return np.eye(3), dict(angle=0, scale=1, tx=0, ty=0, tilt_x=0, tilt_y=0, gain=1, offset=0, blur=0)
    w, h = size
    # A pinhole camera viewing a plane: small roll, pitch/yaw, translation and
    # height change, expressed as a plane-induced homography with known K.
    angle = float(rng.uniform(-5, 5))
    tilt_x, tilt_y = rng.uniform(-2, 2, 2)
    scale = float(rng.uniform(.95, 1.05))
    tx, ty = rng.uniform(-.025, .025, 2) * [w, h]
    rz = cv2.Rodrigues(np.array([0., 0., np.radians(angle)]))[0]
    rotation = cv2.Rodrigues(np.radians([tilt_x, tilt_y, 0]))[0] @ rz
    k = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1.]])
    translation = np.array([tx/w, ty/w, 1/scale-1])
    matrix = k @ (rotation + np.outer(translation, [0, 0, 1])) @ np.linalg.inv(k)
    gain, offset, blur = (float(rng.uniform(.85, 1.15)), float(rng.uniform(-10, 10)), .6) if mode == 'view_lighting' else (1., 0., 0.)
    return matrix/matrix[2, 2], dict(angle=angle, scale=scale, tx=float(tx), ty=float(ty),
                                   tilt_x=float(tilt_x), tilt_y=float(tilt_y), gain=gain, offset=offset, blur=blur)


def diagnose(image, mask, reference, calibration, truth, extractor=extract_features, matcher=match_features):
    first = extractor(image, calibration)
    second = extractor(reference, calibration)
    a, b = matcher(first, second)
    h, keep = fit_homography(calibration.normalized_points(a), calibration.normalized_points(b),
                             2/calibration.fx) if len(a) else (None, None)
    result = dict(features=len(first.points), matches=len(a), alignment=h is not None)
    if h is not None:
        support = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(support, cv2.convexHull(a[keep].astype(np.float32)).astype(np.int32), 1)
        result['mask_inside_source_hull'] = float(np.count_nonzero((mask > 0) & (support > 0))/max(1, np.count_nonzero(mask)))
        pixels = a[keep].reshape(-1, 1, 2).astype(np.float64)
        predicted = cv2.perspectiveTransform(pixels, calibration.matrix @ h @ np.linalg.inv(calibration.matrix))
        expected = cv2.perspectiveTransform(pixels, np.linalg.inv(truth))
        result['median_transform_error_px'] = float(np.median(np.linalg.norm(predicted-expected, axis=2)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260910)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--matcher', choices=['orb', 'lightglue'], default='orb')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(args.seed)
    rng = np.random.default_rng(args.seed)
    dataset = Path('datasets/segmentation/crack_segmentation_dataset/test')
    available = sorted((dataset/'images').glob('*.jpg'))
    selected = rng.choice(available, size=args.count, replace=False)
    store = TemporalStore(args.output/'reference.sqlite')
    extractor, matcher = extract_features, match_features
    model_start = time.monotonic()
    if args.matcher == 'lightglue':
        import torch
        from src.engine.learned_features import LearnedFeatures
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        learned = LearnedFeatures(args.device)
        extractor, matcher = learned.extract, learned.match
    model_load_seconds = time.monotonic()-model_start
    engine = TemporalEngine(store, extractor=extractor, matcher=matcher)
    items, rows, examples = [], [], []
    start = time.monotonic()
    for i, path in enumerate(selected):
        image = cv2.imread(str(path))
        mask_path = dataset/'masks'/path.name
        mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) >= 128).astype(np.uint8)
        height, width = image.shape[:2]
        calibration = CameraCalibration(width, height, width, width, width/2, height/2, 2, 90,
                                        road_roi=(0, 0, width, height))
        # Independent reference enrollment; pool catalog below to test GPS distractors.
        location = Location(40 + i*20/111320, -105, 5)
        video = Video(path, id=f'reference-{i}', route_id=f'enrollment-{i}', calibration=calibration)
        associations = engine.process(video, observation(image, mask, location))
        association = associations[0] if associations else Association('', None, 'no_damage')
        items.append(dict(path=str(path), mask_path=str(mask_path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          location=asdict(location), calibration=asdict(calibration), baseline=asdict(association)))
    with store.db:
        store.db.execute("UPDATE frames SET route_id='synthetic-road'")
    (args.output/'manifest.json').write_text(json.dumps(dict(seed=args.seed, sample_count=args.count,
        method='seeded random sample without replacement; full binary ground-truth mask per image',
        note='References enrolled independently then pooled; GPS spacing 20m gives nearby unrelated candidates.',
        matcher=args.matcher, device=args.device, model_load_seconds=model_load_seconds,
        versions=dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__),
        engine_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                       [Path('src/engine/temporal.py'), Path('src/engine/motion.py')]},
        items=items), indent=2))
    for mode in ('identity', 'view', 'view_lighting', 'wrong_place'):
        for i, item in enumerate(items):
            image = cv2.imread(item['path'])
            mask = (cv2.imread(item['mask_path'], cv2.IMREAD_GRAYSCALE) >= 128).astype(np.uint8)
            height, width = image.shape[:2]
            calibration = CameraCalibration(**item['calibration'])
            matrix, params = transform(rng, (width, height), mode)
            current = cv2.warpPerspective(image, matrix, (width, height))
            current = np.clip(current.astype(float)*params['gain']+params['offset'], 0, 255).astype(np.uint8)
            if params['blur']:
                current = cv2.GaussianBlur(current, (3, 3), params['blur'])
            current_mask = cv2.warpPerspective(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
            if mode in ('view', 'view_lighting'):
                folder = args.output/mode
                folder.mkdir(exist_ok=True)
                cv2.imwrite(str(folder/f'{i:03d}-image.png'), current)
                cv2.imwrite(str(folder/f'{i:03d}-mask.png'), current_mask*255)
            location = Location(**items[(i+args.count//2) % args.count]['location'] if mode == 'wrong_place' else item['location'])
            dx, dy = rng.normal(0, 2, 2)
            location.latitude += dy/111320
            location.longitude += dx/(111320*np.cos(np.radians(location.latitude)))
            video = Video(Path(item['path']), id=f'query-{mode}-{i}', route_id='synthetic-road', calibration=calibration)
            candidates = store.candidates(video, location)
            t = time.monotonic()
            associations = engine.process(video, observation(current, current_mask, location))
            association = associations[0] if associations else Association('', None, 'no_damage')
            expected = item['baseline']['defect_id']
            if not current_mask.any():
                outcome = 'no_damage'
            elif association.defect_id is None:
                outcome = 'unresolved'
            elif association.defect_id == expected:
                outcome = 'correct_id'
            elif association.status == 'matched':
                outcome = 'wrong_id'
            else:
                outcome = 'duplicate_id'
            row = dict(index=i, source=Path(item['path']).name, mode=mode, outcome=outcome,
                       baseline_available=expected is not None, association=asdict(association),
                       candidate_count=len(candidates), expected_candidate=any(r['video_id']==f'reference-{i}' for r in candidates),
                       elapsed_seconds=time.monotonic()-t, gps=asdict(location), transform=matrix.tolist(), parameters=params,
                       retained_mask_pixels=int(current_mask.sum()), original_mask_pixels=int(mask.sum()))
            if mode != 'wrong_place':
                row.update(diagnose(current, current_mask, image, calibration, matrix, extractor, matcher))
            rows.append(row)
            with (args.output/'cases.jsonl').open('a') as output:
                output.write(json.dumps(row)+'\n')
            # Every query is independent: it must not learn from earlier test views.
            with store.db:
                store.db.execute('DELETE FROM observations WHERE video_id=?', (video.id,))
                store.db.execute('DELETE FROM frames WHERE video_id=?', (video.id,))
                store.db.execute('DELETE FROM defects WHERE id NOT IN (SELECT defect_id FROM observations WHERE defect_id IS NOT NULL)')
            if mode == 'view' and len(examples) < 8:
                panels = []
                for picture, labels in ((image, mask), (current, current_mask)):
                    panel = picture.copy()
                    panel[labels > 0] = (.5*panel[labels > 0]+[0, 0, 127]).clip(0,255).astype(np.uint8)
                    panels.append(cv2.resize(panel, (280, 280)))
                panel = np.hstack(panels)
                cv2.putText(panel, f'{i}: {outcome}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 2)
                examples.append(panel)
            if (i+1) % 25 == 0:
                print(mode, i+1, dict(Counter(r['outcome'] for r in rows if r['mode']==mode)), flush=True)
    summary = {mode: dict(Counter(r['outcome'] for r in rows if r['mode']==mode)) for mode in sorted({r['mode'] for r in rows})}
    summary['baseline_available'] = sum(item['baseline']['defect_id'] is not None for item in items)
    summary['query_seconds'] = sum(r['elapsed_seconds'] for r in rows)
    summary['reference_feature_bytes'] = store.db.execute('SELECT SUM(length(features)) FROM frames').fetchone()[0]
    summary['duration_seconds'] = time.monotonic()-start
    summary['alignment'] = {mode: sum(bool(r.get('alignment')) for r in rows if r['mode']==mode) for mode in ('identity','view','view_lighting')}
    (args.output/'results.json').write_text(json.dumps(rows, indent=2))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    if examples:
        cv2.imwrite(str(args.output/'examples.jpg'), np.vstack(examples))
    store.close()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
