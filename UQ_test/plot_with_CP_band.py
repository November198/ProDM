import json
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import pandas as pd
import pickle
import argparse
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix
from timeseries_cp.utils.data_utils import RegressionType
from timeseries_cp.methods.functional_predictor import FunctionalPredictor, ModulationType
from pdn_monitor import build_pdn_model, compute_prodm_scores

#######################
# Data Loading Utils
#######################

def get_all_raw_signals():
    successes = []    
    # Load evaluation log
    with open(eval_log_path, 'r') as file:
        eval_log = json.load(file)
    # Define condition for filtering keys
    cond_func = lambda key: "test/sim_max_reward" in key and 'nv_ob' not in key and 'action' not in key and 'v_ob' not in key
    # Extract data from log
    successes.extend([value[0] for key, value in eval_log.items() if cond_func(key)])
    results = [successes]
    func_to_float = lambda x: float(x.rstrip('.'))
    #### Baselines
    DER_metric = []; STAC_metric = []
    RND_metric = []; CFM_metric = []; NatPN_metric = []; PCA_kmeans_metric = []
    logpO_metric = []; logpZO_metric = []
    STAC_metric.extend([list(map(func_to_float, value[1].split('/'))) for key, value in eval_log.items() if cond_func(key)])
    PCA_kmeans_metric.extend([list(map(func_to_float, value[2].split('/'))) for key, value in eval_log.items() if cond_func(key)])
    logpO_metric.extend([list(map(func_to_float, value[3].split('/'))) for key, value in eval_log.items() if cond_func(key)])
    logpZO_metric.extend([list(map(func_to_float, value[4].split('/'))) for key, value in eval_log.items() if cond_func(key)])
    DER_metric.extend([list(map(func_to_float, value[5].split('/'))) for key, value in eval_log.items() if cond_func(key)])
    NatPN_metric.extend([list(map(func_to_float, value[6].split('/'))) for key, value in eval_log.items() if cond_func(key)])
    CFM_metric.extend([list(map(func_to_float, value[7].split('/'))) for key, value in eval_log.items() if cond_func(key)])
    RND_metric.extend([list(map(func_to_float, value[8].split('/'))) for key, value in eval_log.items() if cond_func(key)])
    baselines = [STAC_metric, PCA_kmeans_metric, logpO_metric, logpZO_metric,
                 DER_metric, NatPN_metric, CFM_metric, RND_metric]
    return results + baselines

#######################
# Plotting Utilities
#######################

def adjust_label(ax, factor=8):
    """Adjust x-axis labels by multiplying with a factor."""
    current_ticks = ax.get_xticks()
    new_ticks = [int(i) for i in current_ticks * factor]
    ax.set_xticklabels(new_ticks)

def plot_on_subfig_traj(ax, log_probs, successes, factor=8):
    """Plot trajectories on a subplot, distinguishing between success and failure.
    Args:
        ax: Matplotlib axis object
        log_probs: List of log probability trajectories
        successes: List of binary success indicators
        factor: Scaling factor for x-axis
    """
    failure_plotted = False
    success_plotted = False
    multiplier = 8
    xaxis = np.arange(len(log_probs[0])) * multiplier
    for log_prob, success in zip(log_probs, successes):
        alpha = 0.1 if success == 1 else 0.5
        color = 'blue' if success else 'red'
        label = 'Success' if success else 'Failure'
        if success and not success_plotted:
            ax.plot(xaxis, log_prob, color=color, label=label, alpha=alpha)
            success_plotted = True
        elif not success and not failure_plotted:
            ax.plot(xaxis, log_prob, color=color, label=label, alpha=alpha)
            failure_plotted = True
        else:
            ax.plot(xaxis, log_prob, color=color, alpha=alpha)
    ax.set_xlabel('Time Step', fontsize=fsize)
    ax.legend(fontsize=fsize-4, loc='upper center', bbox_to_anchor=(0.42, -0.16), ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=fsize-4)
    ax.grid()

def plot_on_subfig_traj_new(ax, log_probs, successes, predictions, plot_TP=False):
    """Plot mean and standard error of trajectories grouped by prediction outcomes.
    Args:
        ax: Matplotlib axis object
        log_probs: List of log probability trajectories
        successes: List of binary success indicators
        predictions: List of binary prediction outcomes
        plot_TP: If True, plot True Positives/Negatives; if False, plot False Positives/Negatives
    """
    fp_log_probs, fn_log_probs, tp_log_probs, tn_log_probs = [], [], [], []
    multiplier = 8
    xaxis = np.arange(len(log_probs[0])) * multiplier
    for log_prob, success, prediction in zip(log_probs, successes, predictions):
        if success == 1 and prediction == 1:
            fp_log_probs.append(log_prob)
        elif success == 0 and prediction == 1:
            tp_log_probs.append(log_prob)
        elif success == 0 and prediction == 0:
            fn_log_probs.append(log_prob)
        elif success == 1 and prediction == 0:
            tn_log_probs.append(log_prob)

    def plot_mean_with_error(ax, log_probs, color, label):
        mean_log_prob = np.mean(log_probs, axis=0)
        std_error_log_prob = np.std(log_probs, axis=0) / np.sqrt(len(log_probs))
        ax.plot(xaxis, mean_log_prob, color=color, label=label)
        ax.fill_between(xaxis, mean_log_prob - std_error_log_prob, mean_log_prob + std_error_log_prob, color=color, alpha=0.3)

    if plot_TP:
        if tp_log_probs:
            plot_mean_with_error(ax, tp_log_probs, 'red', 'TP')
        if tn_log_probs:
            plot_mean_with_error(ax, tn_log_probs, 'blue', 'TN')
    else:
        if fp_log_probs:
            plot_mean_with_error(ax, fp_log_probs, 'blue', 'FP')
        if fn_log_probs:
            plot_mean_with_error(ax, fn_log_probs, 'red', 'FN')

    ax.set_xlabel('Time Step', fontsize=fsize)
    ax.legend(fontsize=fsize-4, loc='upper center', bbox_to_anchor=(0.42, -0.16), ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=fsize-4)
    ax.grid()

def plot_on_subfig_traj_failure(ax, scores, n_idx, small=True):
    """Plot specific failure trajectories with custom styling.
    Args:
        ax: Matplotlib axis object
        scores: List of score trajectories
        n_idx: List of indices to plot
        small: If True, plot smallest failures; if False, plot largest failures
    """
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_idx)))
    fsize = 28
    for i, score in enumerate(scores):
        alpha = 1
        if i not in n_idx:
            continue
        kk = n_idx.index(i) + 1
        midfix = 'smallest' if small else 'largest'
        label_dict = {1: f'{midfix} failure', 2: 'median success'}
        label = label_dict[kk]
        ax.plot(range(len(score)), score, '-o', color=colors[kk-1], label=label, alpha=alpha)
    ax.set_xlabel('Time Step', fontsize=fsize)
    ax.set_ylabel('Score', fontsize=fsize)
    # Custom legend
    ax.legend(fontsize=fsize, loc='upper center', 
            bbox_to_anchor=(0.5, -0.2),  ncol=1,
            title_fontsize=fsize-2)
    ax.tick_params(axis='both', which='major', labelsize=fsize-4)
    adjust_label(ax)

#######################
# Metric Calculations
#######################

def get_metric(y_true, y_pred):
    """Calculate various classification metrics.
    Returns:
        List containing [TPR, TNR, accuracy, accuracy_weighted]
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    accuracy = (tpr + tnr) / 2
    tnr_weight = y_true.sum() / len(y_true)
    tpr_weight = 1 - tnr_weight
    accuracy_weighted = tpr_weight * tpr + tnr_weight * tnr
    return [tpr, tnr, accuracy, accuracy_weighted]

def get_results(metric, lb=True, CPband=True, suffix='', cce_band=False):
    """Process metrics and get detection results with optional time penalty and cumsum.
    Returns:
        List containing [TPR, TNR, accuracy, accuracy_weighted, amount_exceed_ratio,
                        mean_detection_time, detection_time_SE]
    """
    print(f'##### {suffix} metric')
    first_idx_ls, positive_ls, successes_test, _ = get_detection_with_plot(metric, alpha=alpha, lb=lb, CPband=CPband, suffix=suffix, cce_band=cce_band)
    outputs = get_metric(successes_test, positive_ls)
    outputs.append(factor * np.mean(first_idx_ls))
    outputs.append(factor * np.std(first_idx_ls) / np.sqrt(len(first_idx_ls)))
    return outputs

def get_detection_with_plot(log_probs, alpha=0.1, lb=True, CPband=True, suffix='', cce_band=False):
    """Detect anomalies using prediction bands and create visualization plots.
    Returns:
        Tuple containing (first_idx_ls, positive_ls, successes_test, amount_exceed_ratio)
    检测器三态(由 CPband / cce_band 组合决定):
      CPband=True,  cce_band=False -> ① 逐帧 band, 比原始信号
      CPband=False, cce_band=False -> ② CCE: 先 cumsum, 比全局固定平阈值 
      cce_band=True                -> ③ 逐帧累积 band: 先 cumsum, 再比逐帧 conformal band
    """
    assert args.num_train + args.num_cal <= max_tr
    # do_cumsum: 除"纯逐帧 band"(CPband 且非 cce_band)外, 信号都先累积。
    # use_band : 是否走逐帧 conformal band (否则用终点分位的全局平阈值)。
    do_cumsum = (not CPband) or cce_band
    use_band = CPband or cce_band
    cache_key = (type, suffix)
    if cache_key in target_traj_cache:
        target_traj = target_traj_cache[cache_key]
        # The cached band is reusable across ID/OOD calls, but the data split is not:
        # keep the original calibration prefix and evaluate only the held-out suffix.
        log_probs_train = log_probs[:max_tr]
        successes_train = successes[:max_tr]
        log_probs_test = log_probs[max_tr:max_tr + args.num_te]
        successes_test = successes[max_tr:max_tr + args.num_te]
        print(f'Num success in test is {np.sum(successes_test)} out of {len(successes_test)}')
    else:
        log_probs_train = log_probs[:max_tr]; successes_train = successes[:max_tr]
        log_probs_test = log_probs[max_tr:max_tr+args.num_te]; successes_test = successes[max_tr:max_tr+args.num_te]
        log_probs_train = np.array([log_probs_train[i] for i, success in enumerate(successes_train) if success])
        ntr = int(len(log_probs_train) * args.num_train / (args.num_train + args.num_cal))
        ncal = len(log_probs_train) - ntr
        print(f'#### Use {len(log_probs_train)} successful trajectories for calibration')
        predictor = FunctionalPredictor(modulation_type=ModulationType.Tfunc, regression_type=RegressionType.Mean)
        # cce_band(③) 与 CCE(②) 都在累积信号上标定; ①(纯逐帧 band) 用原始信号。
        train_sig = np.array([np.cumsum(val) for val in log_probs_train]) if do_cumsum else log_probs_train
        if use_band:
            print(f'Number of success for mean {ntr} and for band {ncal}'
                  f'{" (on cumsum)" if cce_band else ""}')
            target_traj = predictor.get_one_sided_prediction_band(train_sig[:ntr],  # 成功 rollout 的(累积)时序，拟合逐帧"平均"
            train_sig[-ncal:],   # 成功 rollout，校准逐帧 band
            alpha=alpha,  # 容忍 OOD 比率（square 用 0.025）
            lower_bound=lb  # logpZO 用 lower_bound=False（因为 score 越大越异常，要看 upper band）
            ).flatten()
        else:
            metric_tr = [val[-1] for val in train_sig]   # 终点累积和(train_sig 已 cumsum)
            threshold = np.quantile(metric_tr, 1 - alpha)
            # Repeat for each trajectory
            target_traj = np.repeat(threshold, len(log_probs_train[0]))
        target_traj_cache[cache_key] = target_traj
    if args.num_train < 20 and modify is False and type_orig == 'can':
        raise ValueError('Main evaluation requires the default 300:700 split')
    else:
        if do_cumsum:
            log_probs_test = [np.cumsum(val) for val in log_probs_test]
        print(f'###  Mean of band width: {np.mean(target_traj)}')
        to_plot = min(150, len(log_probs_test))
        rand_idx = np.random.choice(len(log_probs_test), to_plot, replace=False)
        log_probs_test_plt = [log_probs_test[i] for i in rand_idx]
        successes_test_plt = [successes_test[i] for i in rand_idx]

        first_idx_ls = []; positive_ls = []; amount_exceed_success = []; amount_exceed_failure = []
        for log_prob_test, success in zip(log_probs_test, successes_test):
            positive = 0
            for i, log_prob in enumerate(log_prob_test):
                cond = log_prob <= target_traj[i] if lb else log_prob >= target_traj[i]
                if cond:
                    if success < 1:
                        # Record the first index of failure when ground truth is failure
                        first_idx_ls.append(i)
                    positive = 1
                    if success:
                        amount_exceed_success.append(np.abs(log_prob - target_traj[i]))
                    else:
                        amount_exceed_failure.append(np.abs(log_prob - target_traj[i]))
                    break
            positive_ls.append(positive)

        if len(amount_exceed_failure) > 0:
            eps = 1e-5
            amount_exceed_ratio = np.mean(amount_exceed_failure) / (np.mean(amount_exceed_success) + eps)
        else:
            amount_exceed_ratio = 0

        fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
        multiplier = 8; xaxis = np.arange(len(log_probs_test_plt[0])) * multiplier
        func = np.max if lb else np.min
        placeholder_traj = func(log_probs_train) * np.ones_like(target_traj)
        if lb:
            upper, lower = placeholder_traj, target_traj
        else:
            upper, lower = target_traj, placeholder_traj
        for a in ax:
            a.fill_between(xaxis, upper, lower, color='blue', alpha=0.25)
        plot_on_subfig_traj(ax[0], log_probs_test_plt, successes_test_plt)
        positive_ls_plt = [positive_ls[i] for i in rand_idx]
        plot_on_subfig_traj_new(ax[1], log_probs_test_plt, successes_test_plt, positive_ls_plt, plot_TP=True)
        plot_on_subfig_traj_new(ax[2], log_probs_test_plt, successes_test_plt, positive_ls_plt, plot_TP=False)
        for a in ax:
            a.set_ylabel('Score', fontsize=fsize)
            a.set_xlabel('Time Step', fontsize=fsize)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'prediction_band{suffix}.png'), bbox_inches='tight', pad_inches=0.1)
        plt.close('all')
        first_idx_ls = np.array(first_idx_ls); positive_ls = np.array(positive_ls); successes_test = np.array(successes_test)
        return first_idx_ls, positive_ls, 1-successes_test, amount_exceed_ratio

#######################
# Specialized Detection Methods
#######################

def STAC_detect(metric, alpha=0.1):
    """Implement STAC detection method.
    Returns:
        Tuple containing (metric_te, threshold, first_idx_ls, positive_ls, successes_test, amount_exceed_ratio)
    """
    metric_tr = [val for i, val in enumerate(metric) if i < max_tr]
    successes_train = successes[:max_tr]
    metric_tr = np.array([val for val, success in zip(metric_tr, successes_train) if success])
    metric_tr = [np.cumsum(val)[-1] for val in metric_tr]
    threshold = np.quantile(metric_tr, 1 - alpha)
    metric_te = [val for i, val in enumerate(metric) if i >= max_tr]
    successes_test = successes[max_tr:]
    metric_te = [np.cumsum(val) for val in metric_te]
    first_idx_ls = []; positive_ls = []; amount_exceed_success = []; amount_exceed_failure = []
    for metric_te_val, success in zip(metric_te, successes_test):
        positive = 0
        for i, metric_te_ in enumerate(metric_te_val):
            cond = metric_te_ > threshold
            if cond:
                first_idx_ls.append(i)
                positive = 1
                if success:
                    amount_exceed_success.append(np.abs(metric_te_ - threshold))
                else:
                    amount_exceed_failure.append(np.abs(metric_te_ - threshold))
                break
        positive_ls.append(positive)
    first_idx_ls = np.array(first_idx_ls)
    positive_ls = np.array(positive_ls)
    successes_test = np.array(successes_test)
    if len(amount_exceed_failure) > 0:
        eps = 1e-5
        amount_exceed_ratio = np.mean(amount_exceed_failure) / (np.mean(amount_exceed_success) + eps)
    else:
        amount_exceed_ratio = 0
    return metric_te, threshold, first_idx_ls, positive_ls, 1 - successes_test, amount_exceed_ratio

#######################
# Helper Functions
#######################

def get_steps():
    if args.diffusion_policy:
        num_inference_steps = 60
        if type == 'transport':
            num_inference_steps = 70
    else:
        num_inference_steps = 1
        if type == 'toolhang':
            num_inference_steps = 40
    return num_inference_steps

def get_alpha():
    if args.diffusion_policy:
        if 'transport' in type:
            alpha = 0.05
        else:
            alpha = 0.0125
    else:
        if 'transport' in type:
            alpha = 0.01
        else:
            alpha = 0.025
    return alpha

def str2bool(value):
    """Convert string to boolean for argparse."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_percentile_value(data, percentile):
    """Get value at specified percentile from sorted data."""
    # Sort the data based on the second values in the tuples
    sorted_data = sorted(data, key=lambda x: x[1])
    # Calculate the index for the percentile
    index = math.ceil((percentile / 100) * len(sorted_data)) - 1
    # Get the first value of the tuple at the calculated index
    return sorted_data[index][0], sorted_data[index][1]    

def plot_failure(scores, small=True):
    """Create failure analysis plot.
    Returns:
        Matplotlib figure object
    """
    failure_scores = [(i, sum(sublist)) for i, (sublist, success) in enumerate(zip(scores, successes)) if success == 0]
    success_scores = [(i, sum(sublist)) for i, (sublist, success) in enumerate(zip(scores, successes)) if success == 1]
    val = 0 if small else 100
    n_idx_failure, _ = get_percentile_value(failure_scores, val)
    n_idx_success, _ = get_percentile_value(success_scores, 50)
    n_idx = [n_idx_failure, n_idx_success]
    print(f'Rollout index for score: {n_idx_failure}')
    func = np.argmin if small else np.argmax
    sccore_ = np.array(scores[n_idx_failure])
    diff = sccore_[1:] - sccore_[:-1]
    print(f'Steepest change at t = {8*(func(diff)+1)}')
    size = 6
    fig, ax = plt.subplots(figsize=(size, size))
    plot_on_subfig_traj_failure(ax, scores, n_idx, small = small)
    fig.tight_layout()
    return fig

#######################
# Main Script
#######################

parser = ArgumentParser()
parser.add_argument('--diffusion_policy', action='store_true', help='Whether to use diffusion policy')
parser.add_argument('--num_train', type=int, default=300, help='Number of training samples')
parser.add_argument('--num_cal', type=int, default=700, help='Number of testing samples')
parser.add_argument('--num_te', type=int, default=1000, help='Number of testing samples')
args = parser.parse_args()
num_tr_old, num_cal_old, num_te_old = args.num_train, args.num_cal, args.num_te

result_dict = {}; separate = False
target_traj_cache = {}
prodm_model_cache = {}  # Fitted on ID D_fit and reused on OOD.
rows = []
for modify in [False, True]:
    # tool_hang 只有 n=200 < 默认 num_train+num_cal+num_te(=2000),下面在加载后
    # 按该任务实际 rollout 数自适应缩放 cal/test 切分,避免 successes[max_tr:] 除零。
    for type_orig in ['square', 'transport', 'tool_hang', 'can']:
        args.num_train, args.num_cal, args.num_te = num_tr_old, num_cal_old, num_te_old
        max_tr = args.num_train + args.num_cal
        type = type_orig if type_orig != 'tool_hang' else 'toolhang'
        rows.append(f'{type}_ID' if modify is False else f'{type}_OOD')
        num_inference_steps = get_steps()
        print(f'##### Task type: {type}, Num inference steps: {num_inference_steps}, Environment modified: {modify} #####')
        policy = 'diffusion' if args.diffusion_policy else 'flow'
        suffix = '_abs' if type == 'toolhang' else ''
        modify_suffix = '_modify' if modify else ''
        data_dir = os.path.join(f"../data/outputs/train_{policy}_unet_visual_{type_orig}_image{suffix}/final_eval", f'steps_{num_inference_steps}{modify_suffix}')
        eval_log_path = os.path.join(data_dir, f'eval_log_steps_{num_inference_steps}.json')
        print(f'Loading from {eval_log_path}')
        if not os.path.exists(eval_log_path):
            print(f"File {eval_log_path} does not exist!")
            continue
        output_dir = os.path.join('generated', policy, type, 'OOD' if modify else 'ID')
        os.makedirs(output_dir, exist_ok=True)
        everything = get_all_raw_signals()
        cutoff = 7; fsize = 30
        successes = everything[0]
        # 小样本任务(如 tool_hang n=200)自适应:默认要 2000 条(cal 1000 + test 1000),
        # 不够时按实际 rollout 数等比缩放,保持 ~50/50 的 cal/test 划分和 30:70 的
        # train:cal 比例,使 calibration 与 test 都非空(否则 successes[max_tr:] 除零)。
        N_roll = len(successes)
        if N_roll < num_tr_old + num_cal_old + num_te_old:
            cal = max(2, N_roll // 2)
            args.num_train = max(1, int(round(cal * num_tr_old / (num_tr_old + num_cal_old))))
            args.num_cal = max(1, cal - args.num_train)
            max_tr = args.num_train + args.num_cal
            args.num_te = N_roll - max_tr
            print(f'[small-N] {type}: N={N_roll} -> num_train={args.num_train}, '
                  f'num_cal={args.num_cal}, num_te={args.num_te}, max_tr={max_tr} '
                  f'(CI 偏宽,仅供参考)')
        STAC_metric, PCA_kmeans_metric, logpO_metric, logpZO_metric, DER_metric, NatPN_metric, CFM_metric, RND_metric = everything[1:]
        print(f"Entire Success: {sum(successes)}, Failure: {len(successes) - sum(successes)}, Percentage: {sum(successes)/len(successes)}")
        print(f"Only on test: Success: {sum(successes[max_tr:])}, Failure: {len(successes[max_tr:]) - sum(successes[max_tr:])}, Percentage: {sum(successes[max_tr:])/len(successes[max_tr:])}")

        import torch
        cache_path = os.path.join(data_dir, 'global_cond_cache.pt')
        cache = torch.load(cache_path, map_location='cpu')
        embeddings = [np.asarray(g, dtype=np.float32) for g in cache['global_cond']]
        success_mask = np.asarray(successes) >= 1
        if len(embeddings) != len(success_mask) or not np.array_equal(np.asarray(cache['max_reward']) >= 1, success_mask):
            raise ValueError('Embedding cache and log labels differ')
        if not modify:
            successful_ids = np.flatnonzero(success_mask[:max_tr])
            n_fit = int(len(successful_ids) * args.num_train / (args.num_train + args.num_cal))
            if not 0 < n_fit < len(successful_ids):
                raise ValueError('Insufficient successful ID fit/cal data')
            fit_ids = successful_ids[:n_fit]
            prodm_model_cache[type] = build_pdn_model(
                [embeddings[i] for i in fit_ids], [logpZO_metric[i] for i in fit_ids])
        if type not in prodm_model_cache:
            raise ValueError('ID model must be fitted before OOD evaluation')
        prodm_metric = compute_prodm_scores(prodm_model_cache[type], embeddings, logpZO_metric)

        ################## Plot the conformal prediction bands
        alpha = get_alpha()
        factor = 8   
        ### Compare with STAC:
        for metric_name, metric in [("STAC", STAC_metric)]:
            if len(metric) > 0:
                metric_te, threshold, first_idx_ls, positive_ls, successes_test, amount_exceed_ratio = STAC_detect(metric, alpha)
                outputs = get_metric(successes_test, positive_ls)
                # outputs.append(amount_exceed_ratio)
                outputs.append(factor * np.mean(first_idx_ls))
                outputs.append(factor * np.std(first_idx_ls) / np.sqrt(len(first_idx_ls)))

                result_dict[f'{type}_{metric_name}_OOD:{modify}'] = outputs
                fig, ax = plt.subplots(1, 3, figsize=(18, 6))
                to_plot = min(150, len(metric_te))
                rand_idx = np.random.choice(len(metric_te), to_plot, replace=False)
                metric_te_plt = [metric_te[i] for i in rand_idx]
                successes_test_plt = [successes_test[i] for i in rand_idx]
                plot_on_subfig_traj(ax[0], metric_te_plt, successes_test_plt)
                positive_ls_plt = [positive_ls[i] for i in rand_idx]
                plot_on_subfig_traj_new(ax[1], metric_te_plt, successes_test_plt, positive_ls_plt, plot_TP=True)
                plot_on_subfig_traj_new(ax[2], metric_te_plt, successes_test_plt, positive_ls_plt, plot_TP=False)
                for a in ax:
                    a.set_ylabel('Cumulative divergence', fontsize=fsize)
                    a.set_xlabel('Time Step', fontsize=fsize)
                    multiplier = 8
                    xaxis = np.arange(len(metric_te_plt[0])) * multiplier
                    a.fill_between(xaxis, threshold, 0, color='blue', alpha=0.25)
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, f'{metric_name}_plot.png'), bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)
        
        ### Compare with other metrics
        result_dict[f'{type}_PCA-kmeans_OOD:{modify}'] = get_results(PCA_kmeans_metric, lb = False, suffix = '_PCA-kmeans')
        logpO_metric = [-np.array(val) for val in logpO_metric]
        result_dict[f'{type}_logpO_OOD:{modify}'] = get_results(logpO_metric, lb = False, suffix = '_logpO')
        result_dict[f'{type}_logpZO-Ot_OOD:{modify}'] = get_results(logpZO_metric, lb = False, suffix = '_logpZO-Ot')
        result_dict[f'{type}_DER_OOD:{modify}'] = get_results(DER_metric, lb = False, suffix = '_DER')
        NatPN_metric = [-np.array(val) for val in NatPN_metric]
        result_dict[f'{type}_NatPN_OOD:{modify}'] = get_results(NatPN_metric, lb = False, suffix = '_NatPN')
        result_dict[f'{type}_CFM_OOD:{modify}'] = get_results(CFM_metric, lb = False, suffix = '_CFM')        
        result_dict[f'{type}_RND-Ot+At_OOD:{modify}'] = get_results(RND_metric, lb = False, suffix = '_RND-Ot+At')
        result_dict[f'{type}_ProDM_OOD:{modify}'] = get_results(
            prodm_metric, lb=False, CPband=False, suffix='_ProDM', cce_band=True)


# Organize the data into a structured format
metrics = ['TPR', 'TNR', 'Accuracy', 'Accuracy weighted', 'Detect Time', 'Detect Time SE']
# Store everything for use later
# 原版 (注释保留):
# methods = ['STAC', 'PCA-kmeans', 'logpO', 'logpZO-Ot', 'DER', 'NatPN', 'CFM', 'RND-Ot+At']
methods = ['STAC', 'PCA-kmeans', 'logpO', 'logpZO-Ot', 'DER', 'NatPN', 'CFM', 'RND-Ot+At', 'ProDM']
index = pd.MultiIndex.from_product([rows, metrics])
columns = pd.Index(methods, name='Method')
# Create a DataFrame with NaN values
df = pd.DataFrame(np.nan, index=index, columns=columns)
# Populate the DataFrame
for key, value in result_dict.items():
    row_method, ood = key.split('_OOD:')
    row, method = row_method.split('_')
    row = row + ('_ID' if ood == 'False' else '_OOD')
    for metric, val in zip(metrics, value):
        df.loc[(row, metric), method] = val
print(df.round(2))
dir_now = 'logging'
os.makedirs(dir_now, exist_ok=True)
pickle.dump(df, open(f'{dir_now}/metrics_failure_detection_tr{args.num_train}_cal{args.num_cal}_te{args.num_te}_DP{args.diffusion_policy}.pkl', 'wb'))
