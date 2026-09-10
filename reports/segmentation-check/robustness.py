"""Audit exact split leakage and bootstrap the paired clean-sample IoU gap."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
DATA=ROOT/'datasets/segmentation/crack_segmentation_dataset'
seen={hashlib.sha256(p.read_bytes()).hexdigest() for split in ('train','val') for p in (DATA/split/'images').glob('*.jpg')}
sample=json.loads((OUT/'sample.json').read_text())['segmentation']
excluded=[Path(p).name for p in sample if hashlib.sha256((ROOT/p).read_bytes()).hexdigest() in seen]
rows=list(csv.DictReader((OUT/'pixel_results.csv').open()))
clean=[]
arrays=[]
for name in ('packaged','local_epoch1'):
    for threshold in (0.1,0.25,0.5):
        rr=[r for r in rows if r['model']==name and float(r['threshold'])==threshold and r['image'] not in excluded]
        vals=np.array([[int(r[k]) for k in ('tp','fp','fn')] for r in rr])
        tp,fp,fn=vals.sum(axis=0)
        clean.append(dict(model=name,threshold=threshold,images=len(rr),positive_images=sum(int(r['gt_pixels'])>0 for r in rr),pixel_iou=float(tp/(tp+fp+fn)),pixel_precision=float(tp/(tp+fp)),pixel_recall=float(tp/(tp+fn))))
        if threshold==0.25: arrays.append(vals)
rng=np.random.default_rng(42)
indices=rng.integers(0,len(arrays[0]),size=(2000,len(arrays[0])))
scores=[]
for a in arrays:
    totals=a[indices].sum(axis=1)
    scores.append(totals[:,0]/totals.sum(axis=1))
result=dict(excluded_images=excluded,clean_summary=clean,paired_iou_gap_95_percentile_interval=np.quantile(scores[0]-scores[1],[0.025,0.975]).tolist(),bootstrap_replicates=2000)
(OUT/'robustness.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
