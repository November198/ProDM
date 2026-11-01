"""Rollout-level upper conformal band used by the main ProDM experiment."""
import numpy as np


def cumulative_band(fit_scores, cal_scores, alpha):
    if not 0 < alpha < 1 or len(fit_scores)<1 or len(cal_scores)<1:
        raise ValueError('Need nonempty disjoint fit/cal sets and alpha in (0,1)')
    training_data=np.asarray([np.cumsum(x) for x in fit_scores])
    calibration_data=np.asarray([np.cumsum(x) for x in cal_scores])
    if training_data.ndim!=2 or calibration_data.ndim!=2 or training_data.shape[1]!=calibration_data.shape[1]:
        raise ValueError('Main protocol requires a common fixed score horizon')
    prediction_trajectory=np.mean(training_data,axis=0,keepdims=True)
    train_size=training_data.shape[0]
    eps=1e-8
    if int(np.ceil((train_size+1)*(1-alpha)))>train_size:
        modulation_trajectory=np.max(np.abs(training_data-prediction_trajectory),axis=0,keepdims=True)+eps
    else:
        gamma=np.sort(np.max(np.abs(training_data-prediction_trajectory),axis=1))[int(np.ceil((train_size+1)*(1-alpha)))-1]
        modulation_trajectory=np.max(np.abs(training_data-prediction_trajectory)[np.max(np.abs(training_data-prediction_trajectory),axis=1)<=gamma],axis=0,keepdims=True)+eps
    scores=np.sort(np.asarray([np.max((x-prediction_trajectory)/modulation_trajectory) for x in calibration_data],dtype=float))
    k=int(np.ceil((len(scores)+1)*(1-alpha)))
    if not 1<=k<=len(scores):
        raise ValueError('Insufficient calibration samples for the requested alpha')
    return (prediction_trajectory+float(scores[k-1])*modulation_trajectory).flatten()
