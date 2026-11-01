"""Main ProDM evaluation. Fits on ID, then reuses the model/band for ID and OOD."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
try:
    from .pdn_monitor import build_pdn_model, compute_prodm_scores
    from .conformal import cumulative_band
except ImportError:
    from pdn_monitor import build_pdn_model, compute_prodm_scores
    from conformal import cumulative_band

FACTOR=8
TASKS={'square':'square','can':'can','transport':'transport','toolhang':'tool_hang'}


def settings(policy,task):
    if policy=='diffusion':
        return (70,0.05) if task=='transport' else (60,0.0125)
    return (40 if task=='toolhang' else 1,0.01 if task=='transport' else 0.025)


def load_domain(data_root,policy,task,steps,ood=False):
    name=TASKS[task]
    suffix='_abs' if name=='tool_hang' else ''
    directory=Path(data_root)/'outputs'/f'train_{policy}_unet_visual_{name}_image{suffix}'/'final_eval'/f'steps_{steps}{"_modify" if ood else ""}'
    with (directory/f'eval_log_steps_{steps}.json').open() as stream:
        log=json.load(stream)
    entries=[v for k,v in log.items() if 'test/sim_max_reward' in k and not any(x in k for x in ['nv_ob','action','v_ob'])]
    success=np.asarray([float(v['max_reward'] if isinstance(v,dict) else v[0])>=1 for v in entries],dtype=bool)
    density=[np.asarray(v['density'] if isinstance(v,dict) else [float(x.rstrip('.')) for x in v[4].split('/')],dtype=float) for v in entries]
    cache=torch.load(directory/'global_cond_cache.pt',map_location='cpu')
    obs=[np.asarray(x,dtype=np.float32) for x in cache['global_cond']]
    if len(obs)!=len(success) or not np.array_equal(np.asarray(cache['max_reward'])>=1,success):
        raise ValueError(f'Cache and evaluation log do not align: {directory}')
    if any(not np.isfinite(x).all() for x in obs+density):
        raise ValueError(f'Nonfinite embeddings or density: {directory}')
    return obs,density,success


class ProDMDetector:
    """Fit/calibrate once on the ID prefix; no OOD refitting."""
    def __init__(self,alpha=0.025):
        self.alpha=alpha
        self.model=None
        self.band=None

    def fit(self,id_obs,id_density,id_success):
        if min(len(id_obs),len(id_density),len(id_success))<1000:
            raise ValueError('Main protocol needs a 1000-rollout ID monitoring prefix')
        ids=np.flatnonzero(np.asarray(id_success[:1000],dtype=bool))
        nfit=int(len(ids)*300/1000)
        if not 0<nfit<len(ids):
            raise ValueError('Insufficient successful ID fitting/calibration data')
        self.fit_ids,self.cal_ids=ids[:nfit],ids[nfit:]
        self.model=build_pdn_model([id_obs[i] for i in self.fit_ids],[id_density[i] for i in self.fit_ids])
        # Only the ID prefix is scored for fitting and conformal calibration.
        scores=compute_prodm_scores(self.model,id_obs[:1000],id_density[:1000])
        self.band=cumulative_band([scores[i] for i in self.fit_ids],[scores[i] for i in self.cal_ids],self.alpha)
        return self

    def evaluate(self,obs,density,success):
        if self.band is None:
            raise ValueError('Fit on ID before evaluation')
        if min(len(obs),len(density),len(success))<2000:
            raise ValueError('Main evaluation requires indices [1000:2000]')
        scores=compute_prodm_scores(self.model,obs[1000:2000],density[1000:2000])
        metrics=evaluate(scores,success[1000:2000],self.band,0,1000)
        frac=metrics['n_failure']/metrics['n_test']
        metrics.update(FPR=1-metrics['TNR'],WeightedAcc=(1-frac)*metrics['TPR']+frac*metrics['TNR'])
        return metrics


def evaluate(signals, success, band, start, stop):
    indices = range(start, min(stop, len(success)))
    labels, predictions, detected_failure_times = [], [], []
    for i in indices:
        cumulative = np.cumsum(signals[i])
        if len(cumulative) != len(band):
            local_band = band[:len(cumulative)]
        else:
            local_band = band
        hits = np.flatnonzero(cumulative >= local_band)
        detected = bool(len(hits))
        labels.append(int(not success[i]))
        predictions.append(int(detected))
        if detected and not success[i]:
            detected_failure_times.append(int(hits[0]))
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if tp + fn else np.nan
    tnr = tn / (tn + fp) if tn + fp else np.nan
    dt = FACTOR * np.mean(detected_failure_times) if detected_failure_times else np.nan
    dt_se = (FACTOR * np.std(detected_failure_times) / np.sqrt(len(detected_failure_times))
             if detected_failure_times else np.nan)
    return {
        "TPR": tpr,
        "TNR": tnr,
        "BalAcc": (tpr + tnr) / 2,
        "DetectTime": dt,
        "DetectTimeSE": dt_se,
        "n_test": len(labels),
        "n_failure": int(sum(labels)),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    parser.add_argument('--policy',choices=['flow','diffusion','both'],default='both')
    parser.add_argument('--tasks',nargs='+',choices=list(TASKS),default=list(TASKS))
    parser.add_argument('--output',type=Path,help='Optional new CSV file; existing files are refused')
    args=parser.parse_args()
    if args.output and args.output.exists():
        parser.error('Output already exists; choose a new path')
    rows=[]
    for policy in (['flow','diffusion'] if args.policy=='both' else [args.policy]):
        for task in args.tasks:
            steps,alpha=settings(policy,task)
            obs,density,success=load_domain(args.data_root,policy,task,steps)
            detector=ProDMDetector(alpha).fit(obs,density,success)
            print(f'{policy}/{task}: fit={len(detector.fit_ids)}, cal={len(detector.cal_ids)}, alpha={alpha}',flush=True)
            rows.append(dict(policy=policy,task=task,domain='ID',alpha=alpha,**detector.evaluate(obs,density,success)))
            del obs,density,success
            obs,density,success=load_domain(args.data_root,policy,task,steps,ood=True)
            rows.append(dict(policy=policy,task=task,domain='OOD',alpha=alpha,**detector.evaluate(obs,density,success)))
            for row in rows[-2:]:
                print(json.dumps(row),flush=True)
            del obs,density,success,detector
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)


if __name__=='__main__':
    main()
