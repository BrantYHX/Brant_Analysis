import numpy as np
import analysis_functions as af
from scipy.ndimage import gaussian_filter1d

def zscore_session(vec):
    """Z-score using whole-session mean/std."""
    v = np.asarray(vec, dtype=float)
    mu = np.nanmean(v)
    sd = np.nanstd(v) + 1e-9
    return (v - mu) / sd, mu, sd

def zscore_rows(mat, baseline_frames=None):
    """
    Z-score each trial (row). If baseline_frames is provided, compute mean/std
    from mat[:, :baseline_frames]. Otherwise use the whole row.
    """
    mat = np.asarray(mat, dtype=float)
    if baseline_frames is None:
        mu = np.nanmean(mat, axis=1, keepdims=True)
        sd = np.nanstd(mat, axis=1, keepdims=True) + 1e-9
    else:
        mu = np.nanmean(mat[:, :baseline_frames], axis=1, keepdims=True)
        sd = np.nanstd(mat[:, :baseline_frames], axis=1, keepdims=True) + 1e-9
    return (mat - mu) / sd

# def process_pupil(data_ani):
#     data_ani['pupil_diam'] = {}
#     data_ani['pupil_diam_rel'] = {}
#     for gr, trials in data_ani['grating_indices'].items():
#         mat = np.vstack([data_ani['pupil'].iloc[rows].to_numpy() for rows in trials])
#         data_ani['pupil_diam'][gr] = mat
#         base = np.nanmean(mat[:, :10], axis=1, keepdims=True)
#         data_ani['pupil_diam_rel'][gr] = mat / (base + 1e-9)
#     mat = np.vstack([data_ani['pupil'].iloc[rows].to_numpy() for rows in data_ani['reward_indices']])
#     data_ani['pupil_diam']['rew'] = mat
#     base = np.nanmean(mat[:, :10], axis=1, keepdims=True)
#     data_ani['pupil_diam_rel']['rew'] = mat / (base + 1e-9)

#     return data_ani

# def process_speed(data_ani, speed_key="speed", baseline_frames=10):
#     s = data_ani[speed_key].to_numpy()
#     data_ani["speed"] = {}
#     data_ani["speed_rel"] = {}
#     for gr, trials in data_ani["grating_indices"].items():
#         if len(trials) == 0:
#             data_ani["speed"][gr] = np.empty((0, 0))
#             data_ani["speed_rel"][gr] = np.empty((0, 0))
#             continue

#         mat = np.vstack([s[rows] for rows in trials])
#         data_ani["speed"][gr] = mat
#         base = np.nanmean(mat[:, :baseline_frames], axis=1, keepdims=True)
#         data_ani["speed_rel"][gr] = mat / (base + 1e-9)

#     if "reward_indices" in data_ani and data_ani["reward_indices"] is not None and len(data_ani["reward_indices"]) > 0:
#         mat = np.vstack([s[rows] for rows in data_ani["reward_indices"]])
#         data_ani["speed"]["rew"] = mat

#         base = np.nanmean(mat[:, :baseline_frames], axis=1, keepdims=True)
#         data_ani["speed_rel"]["rew"] = mat / (base + 1e-9)

#     return data_ani

def process_pupil(data_ani, baseline_frames=5):
    """
    Outputs:
      data_ani['pupil_rel'][cond] : trials x time (relative pupil diameter)
    """
    p = data_ani["pupil"].to_numpy().astype(float)
    p_sm = gaussian_filter1d(p, sigma=0.7)
    session_baseline = np.nanpercentile(p_sm, 50)
    p_bs = p_sm - session_baseline
    data_ani["pupil_rel"] = {}

    for gr, trials in data_ani["grating_indices"].items():
        # mat = np.vstack([p[rows] for rows in trials])

        # baseline = np.nanmean(mat[:, :baseline_frames], axis=1, keepdims=True)
        # data_ani["pupil_rel"][gr] = mat / (baseline + 1e-9)
        mat = np.vstack([p_bs[rows] for rows in trials])
        data_ani["pupil_rel"][gr] = mat
    # reward
    if "reward_indices" in data_ani and len(data_ani["reward_indices"]) > 0:
        mat = np.vstack([p[rows] for rows in data_ani["reward_indices"]])
        baseline = np.nanmean(mat[:, :baseline_frames], axis=1, keepdims=True)
        data_ani["pupil_rel"]["rew"] = mat / (baseline + 1e-9)

    return data_ani

def process_speed(data_ani, speed_key="speed", baseline_frames=10):
    s = np.asarray(data_ani[speed_key], dtype=float)  # works for Series or ndarray
    s_sm = gaussian_filter1d(s, sigma=0.7)
    session_baseline = np.nanpercentile(s_sm, 50)
    s_bs = s_sm - session_baseline
    data_ani["speed_cm_s"] = {}
    data_ani["speed_delta_cm_s"] = {}

    for gr, trials in data_ani["grating_indices"].items():
        mat = np.vstack([s_bs[rows] for rows in trials])  # m/s -> cm/s
        data_ani["speed_cm_s"][gr] = mat

        base = np.nanmean(mat[:, :baseline_frames], axis=1, keepdims=True)
        data_ani["speed_delta_cm_s"][gr] = mat - base

    if "reward_indices" in data_ani and data_ani["reward_indices"]:
        mat = np.vstack([s[rows] for rows in data_ani["reward_indices"]]) 
        data_ani["speed_cm_s"]["rew"] = mat

        base = np.nanmean(mat[:, :baseline_frames], axis=1, keepdims=True)
        data_ani["speed_delta_cm_s"]["rew"] = mat - base

    return data_ani



def get_unpred_grats(data_ani):
    return {
        gr: data_ani['unpred_trials'][gr]
        for gr in data_ani['activity']
        if gr.startswith("gr_") and gr in data_ani['unpred_trials'] and len(data_ani['unpred_trials'][gr]) > 0
    }

def init_trial_blocks(unpred_gratings, max_tr=4):
    blo1, blo2 = {}, {}
    for ind, gr in enumerate(unpred_gratings):
        blo1[f"X{ind}"] = unpred_gratings[gr][:max_tr]
        blo2[f"X{ind}"] = unpred_gratings[gr][15:15+max_tr]
        # blo2[f"X{ind}"] = unpred_gratings[gr][-20:-20+max_tr]
    return blo1, blo2

def init_trial_blocks_silencing(unpred_gratings, max_tr=4):
    blo1, blo2 = {}, {}
    for ind, gr in enumerate(unpred_gratings):
        blo1[f"X{ind}"] = unpred_gratings[gr][5:5+max_tr]
        blo2[f"X{ind}"] = unpred_gratings[gr][-20:-20+max_tr]
    return blo1, blo2

def sparse_pred_trials(data_ani, unpred_gratings, max_tr, method='simple'):
    pred_trials = []
    if method == 'simple':
        for gr in unpred_gratings:
            gr_trials = [item - 1 for item in data_ani["unpred_trials"][gr][:max_tr] if item - 1 not in data_ani["unpred_trials"]]
            pred_trials.extend(gr_trials)
        btri = np.sort(np.array([t for t in pred_trials if t >= 0])) if pred_trials else np.array([])

    elif method == 'complex':
        pred_trials = list(set([item - 1 for item in data_ani['unpred_trials']['gr_2']][:15] + 
                              [item - 2 for item in data_ani['unpred_trials']['gr_2']][:15] + 
                              list(data_ani['pred_trials'][-15:])))
        pred_trials = [x for x in pred_trials if x not in data_ani['unpred_trials']['gr_2']]
    
    btri = np.sort(np.array([t for t in pred_trials if t >= 0])) if pred_trials else np.array([])
    return btri[:max_tr].tolist()

def get_sig_cells(data_ani, blo1_tris, blo2_tris, mapping, thres, pre_frames, post_frames,method ='ttest'):
    """
    Identify significant neurons for each grating label across two trial blocks.

    Args:
        data_ani:     Single animal's data dict containing 'activity'
        blo1_tris:    Trial indices for block 1, keyed by label
        blo2_tris:    Trial indices for block 2, keyed by label
        mapping:      Dict mapping labels to grating identifiers in activity data
        thres:        Significance threshold (p-value or z-score depending on method)
        pre_frames:   Number of frames before stimulus onset
        post_frames:  Number of frames after stimulus onset
        method:       Statistical test to use: 'ttest', 'bootstrap', or 'threshold'

    Returns:
        sig_cells: Dict mapping each label to a list of significant neuron indices,
                   plus 'all' key containing the union across all labels
    """
    sig_cells = {}
    for label in mapping.keys():
        act_grat = mapping.get(label, label)
        activity = data_ani['activity'][act_grat]

        blo1_trials = blo1_tris.get(label, [])
        blo2_trials = blo2_tris.get(label, [])

        if len(blo1_trials) > 0 or len(blo2_trials) > 0:
            if method == 'ttest':
                sig_neurons_blo1 = af.find_significant_neurons_ttest(activity, blo1_trials, thres, pre_frames, post_frames)[0] if len(blo1_trials) > 0 else []
                sig_neurons_blo2 = af.find_significant_neurons_ttest(activity, blo2_trials, thres, pre_frames, post_frames)[0] if len(blo2_trials) > 0 else []
            elif method == 'bootstrap':
                sig_neurons_blo1 = af.find_significant_neurons_bootstrap(activity, blo1_trials, thres, pre_frames, post_frames)[0] if len(blo1_trials) > 0 else []
                sig_neurons_blo2 = af.find_significant_neurons_bootstrap(activity, blo2_trials, thres, pre_frames, post_frames)[0] if len(blo2_trials) > 0 else []
 
            else: # method == 'threshold'
                sig_neurons_blo1 = af.find_significant_neurons_threshold(activity, blo1_trials, thres, pre_frames, post_frames)[0] if len(blo1_trials) > 0 else []
                sig_neurons_blo2 = af.find_significant_neurons_threshold(activity, blo2_trials, thres, pre_frames, post_frames)[0] if len(blo2_trials) > 0 else []
            sig_cells[label] = list(set(sig_neurons_blo1 + sig_neurons_blo2))
        else:
            sig_cells[label] = []
    
        if sig_cells:
            sig_cells['all'] = list(set().union(*(set(sig_cells[key]) for key in sig_cells.keys() if key != 'all')))

    return sig_cells

def baseline_subtract_pertri(activity_data, baseline_frames):
    baseline_means = np.mean(activity_data[:, :, baseline_frames], axis=2, keepdims=True)
    return activity_data - baseline_means

def bootstrap_test_and_remove(data, test_direction, n_bootstraps, alpha, poststim_frames):
    for ani in range(len(data)):
        h_data_x = data[ani]['activity']['gr_2']
        h_data_a = data[ani]['activity']['gr_1']
        blo1, blo2 = data[ani]['unpred_trials']['gr_2'][:10], data[ani]['unpred_trials']['gr_2'][20:30]
        
        blo1_mean_x = np.mean(np.mean(h_data_x[:, blo1, poststim_frames], axis=2), axis=1)
        blo2_mean_x = np.mean(np.mean(h_data_x[:, blo2, poststim_frames], axis=2), axis=1)
        blo1_mean_a = np.mean(np.mean(h_data_a[:, blo1, poststim_frames], axis=2), axis=1)
        blo2_mean_a = np.mean(np.mean(h_data_a[:, blo2, poststim_frames], axis=2), axis=1)
        
        if test_direction == 'increase':
            observed_diffs = blo2_mean_x - blo1_mean_x
            observed_diffs_a = blo2_mean_a - blo1_mean_a
        else:  # 'decrease'
            observed_diffs = blo1_mean_x - blo2_mean_x
            observed_diffs_a = blo1_mean_a - blo2_mean_a
        
        null_diffs = np.zeros((h_data_x.shape[0], n_bootstraps))
        null_diffs_a = np.zeros((h_data_a.shape[0], n_bootstraps))
        combined_data = np.concatenate((h_data_x[:, blo1, :], h_data_x[:, blo2, :]), axis=1)
        combined_data_a = np.concatenate((h_data_a[:, blo1, :], h_data_a[:, blo2, :]), axis=1)
        
        for i in range(n_bootstraps):
            # bootstrap for gr_2
            shuffled_labels = np.random.permutation(combined_data.shape[1])
            shuffled_blo1 = combined_data[:, shuffled_labels[:len(blo1)], :]
            shuffled_blo2 = combined_data[:, shuffled_labels[len(blo1):], :]
            blo1_shuffle_mean = np.mean(np.mean(shuffled_blo1[:, :, poststim_frames], axis=2), axis=1)
            blo2_shuffle_mean = np.mean(np.mean(shuffled_blo2[:, :, poststim_frames], axis=2), axis=1)
            
            if test_direction == 'increase':
                null_diffs[:, i] = blo2_shuffle_mean - blo1_shuffle_mean
            else:
                null_diffs[:, i] = blo1_shuffle_mean - blo2_shuffle_mean
            
            # bootstrap for gr_1
            shuffled_labels = np.random.permutation(combined_data_a.shape[1])
            shuffled_blo1 = combined_data_a[:, shuffled_labels[:len(blo1)], :]
            shuffled_blo2 = combined_data_a[:, shuffled_labels[len(blo1):], :]
            blo1_shuffle_mean = np.mean(np.mean(shuffled_blo1[:, :, poststim_frames], axis=2), axis=1)
            blo2_shuffle_mean = np.mean(np.mean(shuffled_blo2[:, :, poststim_frames], axis=2), axis=1)
            
            if test_direction == 'increase':
                null_diffs_a[:, i] = blo2_shuffle_mean - blo1_shuffle_mean
            else:
                null_diffs_a[:, i] = blo1_shuffle_mean - blo2_shuffle_mean

        p_values = np.mean(null_diffs >= observed_diffs[:, None], axis=1)
        significant_cells = np.where(p_values < alpha)
        p_values_a = np.mean(null_diffs_a >= observed_diffs_a[:, None], axis=1)
        significant_cells_a = np.where(p_values_a < alpha)
        remove = np.intersect1d(significant_cells[0], significant_cells_a[0])

        indices_to_keep = np.setdiff1d(np.arange(data[ani]['activity']['gr_1'].shape[0]), remove)
        for grat in data[ani]['activity']:
            data[ani]['activity'][grat] = data[ani]['activity'][grat][indices_to_keep]

