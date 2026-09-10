"""Reproduce the checkpoint comparison on a deterministic, bounded test sample."""
import csv
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'datasets/segmentation/crack_segmentation_dataset'
CHECKPOINTS = {
    'packaged': ROOT / 'src/ml/models/weights/segmentation/best.pt',
    'local_epoch1': ROOT / 'src/ml/runs/segmentation/yolo26s-crack-segmentation-6/weights/best.pt',
}


def main():
    torch.set_num_threads(4)
    inventory = {}
    hashes = {}
    for split in ('train', 'val', 'test'):
        files = sorted((DATA / split / 'images').glob('*.jpg'))
        labels = list((DATA / split / 'labels').glob('*.txt'))
        inventory[split] = dict(images=len(files), labels=len(labels), empty_labels=sum(not p.read_text().strip() for p in labels), classes=dict(Counter(l.split()[0] for p in labels for l in p.read_text().splitlines() if l.strip())))
        hashes[split] = {hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    inventory['identical_image_overlap'] = {f'{a}_{b}': len(hashes[a] & hashes[b]) for a, b in [('train', 'test'), ('val', 'test'), ('train', 'val')]}
    all_test = sorted((DATA / 'test/images').glob('*.jpg'))
    selected = sorted(random.Random(42).sample(all_test, 128))
    road = sorted(random.Random(42).sample(sorted((ROOT / 'datasets/detection/RD2022_preprocessed/test/images').glob('*.jpg')), 32))
    (OUT / 'sample.json').write_text(json.dumps({'segmentation': [str(p.relative_to(ROOT)) for p in selected], 'road': [str(p.relative_to(ROOT)) for p in road]}, indent=2))
    metadata, rows, road_rows = {}, [], []
    for name, path in CHECKPOINTS.items():
        model = YOLO(path)
        metadata[name] = dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), names=model.names, task=model.task, parameters=sum(p.numel() for p in model.model.parameters()), **{k:model.ckpt.get(k) for k in ('epoch', 'date', 'version', 'train_metrics', 'train_args')})
        print('MODEL', name, flush=True)
        model.predict(np.zeros((448,448,3), dtype=np.uint8), imgsz=448, device='cpu', verbose=False)
        for i, p in enumerate(selected):
            gt = cv2.imread(str(DATA/'test/masks'/p.name), cv2.IMREAD_GRAYSCALE) > 127
            started = time.perf_counter()
            r = model.predict(str(p), imgsz=448, conf=0.1, retina_masks=True, device='cpu', verbose=False)[0]
            elapsed = time.perf_counter()-started
            for threshold in (0.1, 0.25, 0.5):
                pred = np.zeros(gt.shape, dtype=bool)
                count = 0
                if r.masks is not None:
                    for mask, conf, cls in zip(r.masks.data.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                        if conf >= threshold and int(cls) == 0:
                            pred |= cv2.resize(mask, (gt.shape[1],gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5
                            count += 1
                tp, fp, fn = int((pred&gt).sum()), int((pred&~gt).sum()), int((~pred&gt).sum())
                rows.append(dict(model=name,image=p.name,threshold=threshold,tp=tp,fp=fp,fn=fn,gt_pixels=int(gt.sum()),pred_pixels=int(pred.sum()),instances=count,iou=tp/(tp+fp+fn) if tp+fp+fn else None,seconds=elapsed))
            if i in (0,31,63,95,127): print('TEST',name,i+1,flush=True)
            if i < 6:
                orig = cv2.imread(str(p)); overlay=orig.copy(); overlay[gt] = (0.4*overlay[gt]+0.6*np.array([40,200,40])).astype(np.uint8)
                vis = r.plot(conf=True)
                cv2.imwrite(str(OUT/f'{name}_example_{i}.jpg'),np.hstack([orig,overlay,vis]))
        for size in (448,736):
            for i,p in enumerate(road):
                r=model.predict(str(p),imgsz=size,conf=0.25,retina_masks=True,device='cpu',verbose=False)[0]
                road_rows.append(dict(model=name,image=p.name,imgsz=size,instances=len(r.boxes),cracks=int((r.boxes.cls==0).sum()),potholes=int((r.boxes.cls==1).sum()),inference_ms=r.speed['inference']))
                if size==736 and i<4: cv2.imwrite(str(OUT/f'{name}_road_{i}.jpg'),r.plot())
            print('ROAD',name,size,flush=True)
        (OUT/'progress.json').write_text(json.dumps({'metadata':metadata,'inventory':inventory},indent=2,default=str))
    for filename, records in [('pixel_results.csv',rows),('road_results.csv',road_rows)]:
        with (OUT/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=records[0]); w.writeheader(); w.writerows(records)
    summary=[]
    for name in CHECKPOINTS:
        for threshold in (0.1,0.25,0.5):
            rr=[r for r in rows if r['model']==name and r['threshold']==threshold]
            tp,fp,fn=(sum(r[k] for r in rr) for k in ('tp','fp','fn'))
            positive=[r for r in rr if r['gt_pixels']>0]; negative=[r for r in rr if r['gt_pixels']==0]
            summary.append(dict(model=name,threshold=threshold,images=len(rr),positive_images=len(positive),negative_images=len(negative),pixel_precision=tp/(tp+fp) if tp+fp else 0,pixel_recall=tp/(tp+fn),pixel_iou=tp/(tp+fp+fn),pixel_dice=2*tp/(2*tp+fp+fn),mean_positive_iou=float(np.mean([r['iou'] for r in positive])),positive_images_with_prediction=sum(r['pred_pixels']>0 for r in positive),negative_images_with_prediction=sum(r['pred_pixels']>0 for r in negative),median_seconds=float(np.median([r['seconds'] for r in rr]))))
    result={'inventory':inventory,'metadata':metadata,'summary':summary,'runtime':{'torch':torch.__version__,'device':'cpu','threads':4}}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,default=str))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
