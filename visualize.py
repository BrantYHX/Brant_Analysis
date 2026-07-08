import matplotlib.pyplot as plt
import numpy as np
import analysis_functions as af
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem
from matplotlib import font_manager
import seaborn as sns
import init_vars as iv

def plot_shaded_error(axes, x_vals, data, color='k', label=None, alpha=0.2, ylim=None, title=None,style=None):
    """Plot mean with shaded error bars (std) on given axes."""
    mean_vals = np.nanmean(data, axis=0)
    std_vals = np.nanstd(data, axis=0) / np.sqrt(len(mean_vals))

    if style == 'smooth':
        window_size = 3
        mean_vals = savgol_filter(mean_vals, window_length=window_size, polyorder=1)
        std_vals = savgol_filter(std_vals, window_length=window_size, polyorder=1)

    if style == 'dash':
        linestyle = '--'
    elif style == 'dot':
        linestyle = ':'
    else:
        linestyle = '-'

    axes.plot(x_vals, mean_vals, color=color, label=label, linestyle=linestyle, linewidth=2)
    axes.fill_between(x_vals, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=alpha)
    
    if ylim:
        axes.set_ylim(ylim)
    if label:
        axes.legend()
    if title:
        axes.set_title(title)

def plot_unexp_psth_concat(data, sig_cells, ani_range, conditions, t_frames=None, figsize=(6,6), ylim=None):
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    
    if t_frames is None:
        t_frames = data[ani_range[0]]['activity']['gr_1'].shape[2]
    
    for cond in conditions:
        cond_data = np.concatenate([
            np.mean(data[ani]['activity'][cond['activity']][sig_cells[ani][cond['cells']]][:, data[ani]['unpred_trials'][cond['trials']][cond['slice']], :], axis=1) 
            for ani in ani_range if len(sig_cells[ani][cond['cells']]) > 0 ], axis=0)
        
        plot_shaded_error(axes, range(t_frames), cond_data, color=cond['color'], alpha=0.2, style='smooth', label=cond['label'])
    
    axes.set_ylabel('z-ΔF/F', fontsize=13)
    axes.set_xlabel('Time (s)', fontsize=13)
    axes.legend(fontsize=13, frameon=False, loc='upper right')
    axes.plot([19,19], ylim or [-.2,1], 'darkgrey', linewidth=2.2, linestyle=':')
    if ylim: axes.set_ylim(ylim)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_xticks([11.5, 19, 37])
    axes.set_xticklabels(['-1', '0', '2.4'])
    axes.tick_params(axis='both', which='major', labelsize=13)
    
    return fig, axes

def plot_unexp_psth(data, sig_cells, ani_range, conditions, t_frames=None, figsize=(6,6), ylim=None):
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    
    if t_frames is None:
        t_frames = data[ani_range[0]]['activity']['gr_1'].shape[2]
    
    for cond in conditions:
        animal_averages = []
        for ani in ani_range:
            if len(sig_cells[ani][cond['cells']]) > 0:
                animal_response = np.mean(np.mean(data[ani]['activity'][cond['activity']][sig_cells[ani][cond['cells']]][:, 
                           data[ani]['unpred_trials'][cond['trials']][cond['slice']], :], axis=1),axis=0) 
                animal_averages.append(animal_response)
        
        if animal_averages:
            cond_data = np.array(animal_averages)
            plot_shaded_error(axes, range(t_frames), cond_data, color=cond['color'], alpha=0.2, style='smooth', label=f"{cond['label']} (n={len(animal_averages)} sessions)")
    
    axes.set_ylabel('z-ΔF/F', fontsize=13)
    axes.set_xlabel('Time (s)', fontsize=13)
    axes.legend(fontsize=13, frameon=False, loc='upper right')
    axes.plot([19,19], ylim or [-.2,1], 'darkgrey', linewidth=2, linestyle=':')
    if ylim: axes.set_ylim(ylim)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_xticks([11.5, 19, 37])
    axes.set_xticklabels(['-1', '0', '2.4'])
    axes.tick_params(axis='both', which='major', labelsize=13)
    
    return fig, axes

def plot_psth(data, sig_cells, ani_range, conditions, t_frames=None, figsize=(6,6), ylim=None):
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    
    if t_frames is None:
        t_frames = data[ani_range[0]]['activity']['gr_1'].shape[2]
    
    for cond in conditions:
        cond_data = np.concatenate([
            np.mean(data[ani]['activity'][cond['activity']][sig_cells[ani][cond['cells']]][:, cond['trials'][ani]['gr_2'], :], axis=1) 
            for ani in ani_range if len(sig_cells[ani][cond['cells']]) > 0 ], axis=0)
        
        plot_shaded_error(axes, range(t_frames), cond_data, color=cond['color'], alpha=0.2, style='smooth', label=cond['label'])
    
    axes.set_ylabel('z-ΔF/F', fontsize=13)
    axes.set_xlabel('Time (s)', fontsize=13)
    axes.legend(fontsize=13, frameon=False, loc='upper right')
    axes.plot([19,19], ylim or [-.2,1], 'darkgrey', linewidth=2, linestyle=':')
    if ylim: axes.set_ylim(ylim)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_xticks([11.5, 19, 37])
    axes.set_xticklabels(['-1', '0', '2.4'])
    axes.tick_params(axis='both', which='major', labelsize=13)
    
    return fig, axes

def plot_adapt(data, sig_cells, min_trials, ani_range,poststim_frames, unpred_grat='gr_2', unpred_ind='X0', figsize=(6,6), ylim=None,axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = axes.get_figure()
    x_vals = list(range(min_trials))
    gr1_data = np.concatenate([
        np.mean(data[ani]['activity']['gr_1'][sig_cells[ani]['gr_1']][:, data[ani]['unpred_trials'][unpred_grat][:min_trials], poststim_frames], axis=2)
        for ani in ani_range if len(sig_cells[ani]['X0']) > 0
    ], axis=0)
    
    axes.errorbar(x_vals, np.mean(gr1_data, axis=0),np.std(gr1_data, axis=0) / np.sqrt(len(gr1_data)),fmt='o-', color='grey', label='A responses')
    
    gr2_data = np.concatenate([
        np.mean(data[ani]['activity'][unpred_grat][sig_cells[ani][unpred_ind]][:, data[ani]['unpred_trials'][unpred_grat][:min_trials], poststim_frames], axis=2)
        for ani in ani_range if len(sig_cells[ani][unpred_ind]) > 0
    ], axis=0)
    
    axes.errorbar(x_vals, np.mean(gr2_data, axis=0), np.std(gr2_data, axis=0) / np.sqrt(len(gr2_data)), fmt='o-', color='k', label='X responses')
    
    axes.set_ylabel('z-ΔF/F', fontsize=13)
    axes.set_xlabel('Trial Number', fontsize=13)
    axes.legend(fontsize=13, frameon=False, loc='upper right')
    if ylim: axes.set_ylim(ylim)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_xticks([5, 15, 30])
    axes.set_yticks([0,0.5,1])
    axes.tick_params(axis='both', which='major', labelsize=13)
    
    return fig, axes

def plot_adapt_bin_pad_acrossanis(data, sig_cells, poststim_frames, ani_range, unpred_grat='gr_2', unpred_ind='X0',figsize=(6,6), ylim=[-.1,1], xlim=[-1,30],ax=None):
    bin_size=2
    unpred_grat, unpred_ind = 'gr_2','X0'
    def pad_with_nans(data_array, n_trials_target):
        if len(data_array) < n_trials_target:
            return np.concatenate([data_array, np.full(n_trials_target - len(data_array), np.nan)])
        return data_array
    
    def bin_data(data_array, bin_size):
        if bin_size is None:
            return data_array
        n_bins = len(data_array) // bin_size
        return np.array([np.nanmean(data_array[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)])
    
    max_trials = max([len(data[ani]['unpred_trials']['gr_2']) for ani in ani_range])
    
    all_x0_data, all_a1_data = [], []
    for ani in ani_range:
        x0_data = np.mean(np.mean(data[ani]['activity']['gr_2'][sig_cells[ani]['X0']][:,data[ani]['unpred_trials']['gr_2'],poststim_frames], axis=2), axis=0)
        a1_data = np.mean(np.mean(data[ani]['activity']['gr_1'][sig_cells[ani]['gr_1']][:,data[ani]['unpred_trials']['gr_2'],poststim_frames], axis=2), axis=0)
        
        x0_padded = pad_with_nans(x0_data, max_trials)
        a1_padded = pad_with_nans(a1_data, max_trials)
        x0_binned = bin_data(x0_padded, bin_size)
        a1_binned = bin_data(a1_padded, bin_size)
        
        all_x0_data.append(x0_binned)
        all_a1_data.append(a1_binned)
    
    all_x0_data, all_a1_data = np.array(all_x0_data), np.array(all_a1_data)
    x0_mean, x0_sem = np.nanmean(all_x0_data, axis=0), sem(all_x0_data, nan_policy='omit', axis=0)
    a1_mean, a1_sem = np.nanmean(all_a1_data, axis=0), sem(all_a1_data, nan_policy='omit', axis=0)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()
    x_vals = np.arange(len(x0_mean))
    
    ax.errorbar(x_vals, a1_mean, yerr=a1_sem, fmt='o-', color='lightgray', markersize=7, 
                markerfacecolor='white', markeredgecolor='gray', label=f'A1 (n={len(ani_range)} animals)')
    ax.errorbar(x_vals, x0_mean, yerr=x0_sem, fmt='o-', color='black', markersize=7, 
                markerfacecolor='dimgray', markeredgecolor='black', label=f'X (n={len(ani_range)} animals)')
    
    ax.axvspan(0, 7, color='red', alpha=0.08)
    ax.axvspan(15, 23, color='dodgerblue', alpha=0.08)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    xticks = np.array([0, 7, 15, 23]) 
    xtick_labels = ['1–2', '15–16', '31–32', '47-48']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticks([0, 0.5,1,1.5])
    
    ax.set_xlabel('Trial', fontsize=18, fontname='Arial')
    ax.set_ylabel('Grating response\n(z-scored ΔF/F)', fontsize=18, fontname='Arial')
    
    ax.tick_params(axis='both', direction='out', length=8, width=1.5, labelsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    arial_font = font_manager.FontProperties(family='Arial', size=18)
    # ax.legend(prop=arial_font, frameon=False, loc='center', bbox_to_anchor=(0.55, 0.6))
    
    return fig, ax

def plot_adapt_bin_pad_acrosscells(data, sig_cells, poststim_frames, ani_range,unpred_grat='gr_2', unpred_ind='X0',figsize=(6,6), ylim=[-.1,1], xlim=[-1,30]):
    bin_size = 2
    
    def pad_with_nans(data_array, n_trials_target):
        if len(data_array) < n_trials_target:
            return np.concatenate([data_array, np.full(n_trials_target - len(data_array), np.nan)])
        return data_array
    
    def bin_data(data_array, bin_size):
        if bin_size is None:
            return data_array
        n_bins = len(data_array) // bin_size
        return np.array([np.nanmean(data_array[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)])
    
    max_trials = max([len(data[ani]['unpred_trials'][unpred_grat]) for ani in ani_range])
    all_x0_data, all_a1_data = [], []
    
    for ani in ani_range:
        x0_cells = np.median(data[ani]['activity'][unpred_grat][sig_cells[ani][unpred_ind]][:,data[ani]['unpred_trials'][unpred_grat],poststim_frames], axis=2)
        a1_cells = np.median(data[ani]['activity']['gr_1'][sig_cells[ani]['gr_1']][:,data[ani]['unpred_trials'][unpred_grat],poststim_frames], axis=2)
        
        for cell_idx in range(x0_cells.shape[0]):
            x0_padded = pad_with_nans(x0_cells[cell_idx], max_trials)
            x0_binned = bin_data(x0_padded, bin_size)
            all_x0_data.append(x0_binned)
        
        for cell_idx in range(a1_cells.shape[0]):
            a1_padded = pad_with_nans(a1_cells[cell_idx], max_trials)
            a1_binned = bin_data(a1_padded, bin_size)
            all_a1_data.append(a1_binned)
    
    all_x0_data, all_a1_data = np.array(all_x0_data), np.array(all_a1_data)
    x0_mean, x0_sem = np.nanmean(all_x0_data, axis=0), sem(all_x0_data, nan_policy='omit', axis=0)
    a1_mean, a1_sem = np.nanmean(all_a1_data, axis=0), sem(all_a1_data, nan_policy='omit', axis=0)
    
    fig, ax = plt.subplots(figsize=figsize)
    x_vals = np.arange(len(x0_mean))
    
    ax.errorbar(x_vals, a1_mean, yerr=a1_sem, fmt='o-', color='lightgray', markersize=7,
                markerfacecolor='white', markeredgecolor='gray', label=f'A1 (n={len(all_a1_data)} cells)')
    ax.errorbar(x_vals, x0_mean, yerr=x0_sem, fmt='o-', color='black', markersize=7,
                markerfacecolor='dimgray', markeredgecolor='black', label=f'X (n={len(all_x0_data)} cells)')
    
    ax.axvspan(0, 7, color='red', alpha=0.08)
    ax.axvspan(15, 23, color='dodgerblue', alpha=0.08)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    
    xticks = np.array([0, 7, 15, 23])
    xtick_labels = ['1–2', '15–16', '31–32', '47-48']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticks([0, 0.5, 1, 1.5])
    
    ax.set_xlabel('Trial', fontsize=18, fontname='Arial')
    ax.set_ylabel('Grating response\n(z-scored ΔF/F)', fontsize=18, fontname='Arial')
    ax.tick_params(axis='both', direction='out', length=8, width=1.5, labelsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    arial_font = font_manager.FontProperties(family='Arial', size=18)
    ax.legend(prop=arial_font, frameon=False, loc='center', bbox_to_anchor=(0.55, 0.6))
    
    return fig, ax

def plot_heatmaps(data, ani_range, conditions, poststim_frames, max_tri=15,figsize=(12,6), vmin=-1, vmax=1, sort_by='self',label='axonal bouton'):
    condition_data = []
    for cond in conditions:
        cond_list = []
        for ani in ani_range:
            if cond['type'] == 'predicted':
                trials = iv.sparse_pred_trials(data[ani], data[ani]['unpred_trials'], max_tri, method='complex')
            elif cond['type'] == 'unpred':
                trials = data[ani]['unpred_trials']['gr_2'][:max_tri]
            elif cond['type'] == 'difference':
                early_trials = data[ani]['unpred_trials']['gr_2'][:max_tri]
                late_trials = data[ani]['unpred_trials']['gr_2'][-max_tri:]
                early_data = np.mean(data[ani]['activity'][cond['activity']][:, early_trials, :], axis=1)
                late_data = np.mean(data[ani]['activity'][cond['activity']][:, late_trials, :], axis=1)
                cond_list.append(early_data - late_data)
                continue
            
            if cond['type'] != 'difference':
                cond_list.append(np.mean(data[ani]['activity'][cond['activity']][:, trials, :], axis=1))
        
        condition_data.append(np.concatenate(cond_list, axis=0))
    
    fig, axes = plt.subplots(1, len(conditions), figsize=figsize)
    if len(conditions) == 1:
        axes = [axes]
    
    for i, (cond, vals) in enumerate(zip(conditions, condition_data)):
        ax = axes[i]
        if sort_by == 'self':
            sort_indices = np.argsort(np.mean(vals[:, poststim_frames], axis=1))
        else:
            sort_indices = np.argsort(np.mean(condition_data[sort_by][:, poststim_frames], axis=1))
        sorted_vals = vals[sort_indices, :]
        sns.heatmap(sorted_vals, ax=ax, cmap='coolwarm', cbar=False, vmin=vmin, vmax=vmax)
        ax.set_title(cond['title'], fontsize=18)
        ax.set_xlabel('Time (s)', fontsize=15)
        ax.set_xticks([12.5, 20, 38])
        ax.set_xticklabels(['-1', '0', '2.4'])
        ax.tick_params(axis='x', rotation=0, labelsize=15)
        
        if i == 0:
            ax.set_ylabel(f'{label} # (sorted)', fontsize=15)
            num_neurons = vals.shape[0]
            y_ticks = np.arange(0, num_neurons, 1000)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticks)
        else:
            ax.set_yticks([])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.4)
        ax.spines['left'].set_linewidth(0.4)
    
    plt.tight_layout()
    return fig, axes

def plot_quick_heatmap(data, ani, grating, trials, cells=None, sort_frames=slice(23,30),
                     vmin=-2, vmax=2, figsize=(6,8), difference=False, diff_trials=None,
                     smooth_sigma=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    activity = data[ani]['activity'][grating][cells if cells is not None else slice(None)]

    if difference and diff_trials is not None:
        plot_data = np.mean(activity[:, trials, :], axis=1) - np.mean(activity[:, diff_trials, :], axis=1)
    else:
        plot_data = np.mean(activity[:, trials, :], axis=1)

    if smooth_sigma is not None:
        plot_data = gaussian_filter1d(plot_data, sigma=smooth_sigma, axis=1)

    sort_idx = np.argsort(np.mean(plot_data[:, sort_frames], axis=1))
    sns.heatmap(plot_data[sort_idx], ax=ax, cmap='bwr', cbar=True, vmax=vmax, vmin=vmin)
    ax.set_xticks([11.5, 19, 37])
    ax.set_xticklabels(['-1', '0', '2.4'])
    ax.axvline(19, linestyle='-', color='k')
    ax.set_title(grating)
    y_ticks = np.arange(0, plot_data.shape[0], 25)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    ax.set_ylabel('#')

    return fig, ax

def plot_resp_perct(data, ani_range, prestim_frames, poststim_frames, lab = 'cells',
                              threshold=0.3, max_trials=36, figsize=(6,5),color='k',ax =None):

    def get_responsive_pct(activity, trial_indices, prestim_frames, poststim_frames, threshold):
        pcts = []
        for trial in trial_indices:
            pre = np.mean(activity[:, trial, prestim_frames], axis=1)
            post = np.mean(activity[:, trial, poststim_frames], axis=1)
            responsive = ((post - pre) > threshold) & (post > threshold)
            pcts.append(np.mean(responsive))
        return pcts
    
    avg_gr1, avg_gr2 = [], []
    for ani in ani_range:
        n_trials = min(max_trials, len(data[ani]['unpred_trials']['gr_2']))
        trial_indices = data[ani]['unpred_trials']['gr_2'][:n_trials]
        
        avg_gr1.append(get_responsive_pct(data[ani]['activity']['gr_1'], trial_indices, 
                                         prestim_frames, poststim_frames, threshold))
        avg_gr2.append(get_responsive_pct(data[ani]['activity']['gr_2'], trial_indices, 
                                         prestim_frames, poststim_frames, threshold))
    
    max_len = max(len(arr) for arr in avg_gr1 + avg_gr2)
    avg_gr1 = [arr + [np.nan] * (max_len - len(arr)) for arr in avg_gr1]
    avg_gr2 = [arr + [np.nan] * (max_len - len(arr)) for arr in avg_gr2]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    gr1_stack, gr2_stack = np.stack(avg_gr1), np.stack(avg_gr2)
    
    ax.errorbar(np.arange(max_len), np.nanmean(gr1_stack, axis=0), 
                np.nanstd(gr1_stack, axis=0) / np.sqrt(len(gr1_stack)), 
                fmt='o-', color='lightgrey', label='gr_1')
    ax.errorbar(np.arange(max_len), np.nanmean(gr2_stack, axis=0), 
                np.nanstd(gr2_stack, axis=0) / np.sqrt(len(gr2_stack)), 
                fmt='o-', color=color, label='gr_2')
    
    ax.set_ylabel(f'% of {lab}', fontsize=14)
    ax.set_xlabel('unexpected trial #', fontsize=14)
    ax.set_title(f'average % of resp. {lab} across trials', fontsize=15)
    ax.set_xticks([0, 6, 15, 30])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig, ax

def plot_emerging(data, poststim_frames,ani_range, n_bootstraps=500, alpha=0.05, early_slice=slice(0,10), late_slice=slice(20,30), figsize=(14,5)):
    
    def bootstrap_test(activity, blo1, blo2, poststim_frames, n_bootstraps, alpha):
        blo1_mean = np.mean(np.mean(activity[:, blo1, poststim_frames], axis=2), axis=1)
        blo2_mean = np.mean(np.mean(activity[:, blo2, poststim_frames], axis=2), axis=1)
        observed_diffs = blo2_mean - blo1_mean
        
        combined_data = np.concatenate((activity[:, blo1, :], activity[:, blo2, :]), axis=1)
        null_diffs = np.zeros((activity.shape[0], n_bootstraps))
        
        for i in range(n_bootstraps):
            shuffled_labels = np.random.permutation(combined_data.shape[1])
            shuffled_blo1 = combined_data[:, shuffled_labels[:len(blo1)], :]
            shuffled_blo2 = combined_data[:, shuffled_labels[len(blo1):], :]
            blo1_shuffle = np.mean(np.mean(shuffled_blo1[:, :, poststim_frames], axis=2), axis=1)
            blo2_shuffle = np.mean(np.mean(shuffled_blo2[:, :, poststim_frames], axis=2), axis=1)
            null_diffs[:, i] = blo2_shuffle - blo1_shuffle
        
        p_values = np.mean(null_diffs >= observed_diffs[:, None], axis=1)
        return np.where(p_values < alpha)[0]
    
    min_trials = min([len(data[ani]['unpred_trials']['gr_2']) for ani in range(len(data))])
    emerging_cells_x, emerging_cells_a = [], []
    for ani in ani_range:
        blo1 = data[ani]['unpred_trials']['gr_2'][early_slice]
        blo2 = data[ani]['unpred_trials']['gr_2'][late_slice]
        
        emerging_cells_x.append(bootstrap_test(data[ani]['activity']['gr_2'], blo1, blo2, poststim_frames, n_bootstraps, alpha))
        emerging_cells_a.append(bootstrap_test(data[ani]['activity']['gr_1'], blo1, blo2, poststim_frames, n_bootstraps, alpha))
    
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    # Plot A emerging cells
    a_data = np.stack([np.mean(np.mean(data[ani]['activity']['gr_2'][emerging_cells_a[ani]][:,data[ani]['unpred_trials']['gr_2'][:min_trials],poststim_frames], axis=2),axis=0) for ani in ani_range])
    a_data_gr1 = np.stack([np.mean(np.mean(data[ani]['activity']['gr_1'][emerging_cells_a[ani]][:,data[ani]['unpred_trials']['gr_2'][:min_trials],poststim_frames], axis=2),axis=0) for ani in ani_range])
    
    plot_shaded_error(ax[0], np.arange(min_trials), a_data, color='darkcyan', alpha=0.2, style='smooth')
    plot_shaded_error(ax[0], np.arange(min_trials), a_data_gr1, color='gray', alpha=0.2, style='smooth')
    ax[0].set_title('emerge rois A', fontsize=17)
    
    # Plot X emerging cells  
    x_data = np.stack([np.mean(np.mean(data[ani]['activity']['gr_2'][emerging_cells_x[ani]][:,data[ani]['unpred_trials']['gr_2'][:min_trials],poststim_frames], axis=2),axis=0) for ani in ani_range])
    x_data_gr1 = np.stack([np.mean(np.mean(data[ani]['activity']['gr_1'][emerging_cells_x[ani]][:,data[ani]['unpred_trials']['gr_2'][:min_trials],poststim_frames], axis=2),axis=0) for ani in ani_range])
    
    plot_shaded_error(ax[1], np.arange(min_trials), x_data, color='darkcyan', alpha=0.2, style='smooth')
    plot_shaded_error(ax[1], np.arange(min_trials), x_data_gr1, color='gray', alpha=0.2, style='smooth')
    ax[1].set_title('emerge rois X', fontsize=17)
    
    prop_a = [100 * len(emerging_cells_a[ani]) / data[ani]['activity']['gr_1'].shape[0] for ani in ani_range]
    prop_x = [100 * len(emerging_cells_x[ani]) / data[ani]['activity']['gr_2'].shape[0] for ani in ani_range]
    
    for i in range(len(prop_a)):
        ax[2].plot([0, 1], [prop_a[i], prop_x[i]], 'o-', alpha=0.7, color='k')
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(['A', 'X'])
    ax[2].set_title('proportion of emerging rois')
    for axes in ax[:2]:
        axes.set_xlabel('unpredicted trials')
        axes.set_ylabel('avg z-ΔF/F', fontsize=15)
        axes.set_xlim([0, min_trials])
        axes.set_ylim([-0.5, 1])
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
    
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    
    return fig, ax, emerging_cells_a, emerging_cells_x
