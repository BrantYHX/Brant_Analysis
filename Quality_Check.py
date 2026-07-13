import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

import init_vars as iv
import visualize as viz

FPS = 7.5


def compute_sig_cells(data_ani, thres=0.5, prestim_frames=slice(12, 19), poststim_frames=slice(23, 33),
                       max_tr=6, method='threshold'):
    """
    Significant-cell bookkeeping for a single session (mirrors the per-animal loop in the
    LC_Dreadds notebook): returns sig_cells_ani with keys 'gr_1', 'gr_2', 'X0', 'all'.
    """
    unpred_gratings = iv.get_unpred_grats(data_ani)
    blo1, blo2 = iv.init_trial_blocks(unpred_gratings, max_tr=max_tr)
    btri = iv.sparse_pred_trials(data_ani, unpred_gratings, max_tr, method='simple')

    if len(btri) > 0:
        blo1['gr_2'] = btri
        blo2['gr_2'] = btri
    if 'X0' in blo1:
        blo1['gr_1'] = blo1['X0']
        blo2['gr_1'] = blo2['X0']

    mapping = {'gr_1': 'gr_1', 'gr_2': 'gr_2', 'X0': 'gr_2'}
    return iv.get_sig_cells(data_ani, blo1, blo2, mapping, thres, prestim_frames, poststim_frames, method=method)


def plot_qc_heatmaps(data_ani, poststim_frames, max_tri=16, figsize=(18, 6), vmin=-1, vmax=1, sort_by='self'):
    """
    4-panel heatmap of average per-neuron responses, sorted by post-stim response:
      1) A (gr_1), on trials where gr_2 was unexpectedly replaced by X
      2) B (gr_2), on expected/predicted trials
      3) X (gr_2), first block of unexpected trials (trials 1..max_tri)
      4) X (gr_2), second block of unexpected trials (trials max_tri+1..end)
    """
    unpred_gratings = iv.get_unpred_grats(data_ani)
    unpred_gr2 = data_ani['unpred_trials']['gr_2']
    pred_trials = iv.sparse_pred_trials(data_ani, unpred_gratings, max_tri, method='complex')

    conditions = [
        {'activity': 'gr_1', 'trials': unpred_gr2, 'title': 'A (unexpected trials)'},
        {'activity': 'gr_2', 'trials': pred_trials, 'title': 'B (expected trials)'},
        {'activity': 'gr_2', 'trials': unpred_gr2[:max_tri], 'title': f'X (block 1, trials 1-{max_tri})'},
        {'activity': 'gr_2', 'trials': unpred_gr2[16:16+max_tri], 'title': f'X (block 2)'},
    ]
    condition_data = [np.mean(data_ani['activity'][c['activity']][:, c['trials'], :], axis=1) for c in conditions]

    fig, axes = plt.subplots(1, len(conditions), figsize=figsize)
    for i, (cond, vals) in enumerate(zip(conditions, condition_data)):
        ax = axes[i]
        if sort_by == 'self':
            sort_idx = np.argsort(np.mean(vals[:, poststim_frames], axis=1))
        else:
            sort_idx = np.argsort(np.mean(condition_data[sort_by][:, poststim_frames], axis=1))
        sns.heatmap(vals[sort_idx], ax=ax, cmap='coolwarm', cbar=(i == len(conditions) - 1), vmin=vmin, vmax=vmax)

        ax.set_title(cond['title'], fontsize=13)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_xticks([11.5, 19, 37])
        ax.set_xticklabels(['-1', '0', '2.4'])
        ax.tick_params(axis='x', rotation=0, labelsize=11)

        if i == 0:
            ax.set_ylabel('neuron # (sorted)', fontsize=12)
            n = vals.shape[0]
            y_ticks = np.arange(0, n, max(1, n // 10))
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticks)
        else:
            ax.set_yticks([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig, axes


def plot_qc_psth(data_ani, sig_cells_ani, groups=('gr_1', 'gr_2', 'X0'), max_tri=16, figsize=None, ylim=None):
    """
    For each responsive-cell group in `groups`, PSTH of that population's response to:
      A (gr_1), B expected (gr_2, predicted trials), X unexpected (gr_2, first max_tri unpred trials).
    """
    unpred_gratings = iv.get_unpred_grats(data_ani)
    unpred_gr2 = data_ani['unpred_trials']['gr_2'][:max_tri]
    pred_trials = iv.sparse_pred_trials(data_ani, unpred_gratings, max_tri, method='complex')
    t_frames = data_ani['activity']['gr_1'].shape[2]

    figsize = figsize or (4.5 * len(groups), 4)
    fig, axes = plt.subplots(1, len(groups), figsize=figsize)
    if len(groups) == 1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        cells = sig_cells_ani.get(group, [])
        ax.set_title(f'{group}-responsive (n={len(cells)})', fontsize=13)
        if len(cells) == 0:
            continue

        a_resp = np.mean(data_ani['activity']['gr_1'][cells][:, unpred_gr2, :], axis=1)
        b_resp = np.mean(data_ani['activity']['gr_2'][cells][:, pred_trials, :], axis=1)
        x_resp = np.mean(data_ani['activity']['gr_2'][cells][:, unpred_gr2, :], axis=1)

        viz.plot_shaded_error(ax, range(t_frames), a_resp, color='grey', style='dash', label='A')
        viz.plot_shaded_error(ax, range(t_frames), b_resp, color='steelblue', style='dot', label='B (expected)')
        viz.plot_shaded_error(ax, range(t_frames), x_resp, color='black', label='X (unexpected)')

        ax.axvline(19, linestyle='--', color='k', linewidth=1)
        ax.set_xlabel('Time from onset (s)', fontsize=12)
        ax.set_xticks([11.5, 19, 37])
        ax.set_xticklabels(['-1', '0', '2.4'])
        if ylim:
            ax.set_ylim(ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=9)

    axes[0].set_ylabel('z-ΔF/F', fontsize=12)
    plt.tight_layout()
    return fig, axes


def plot_qc_adaptation(data_ani, sig_cells_ani, poststim_frames, max_trials=None, figsize=(5, 4), ax=None):
    """Adaptation of gr_1- and X0-responsive neurons' response amplitude over the unexpected-trial sequence."""
    unpred_gr2 = data_ani['unpred_trials']['gr_2']
    n_trials = len(unpred_gr2)
    trials = unpred_gr2[:n_trials]
    x_vals = list(range(n_trials))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    a_cells = sig_cells_ani.get('gr_1', [])
    if len(a_cells) > 0:
        a_data = np.mean(data_ani['activity']['gr_1'][a_cells][:, trials, poststim_frames], axis=2)
        ax.errorbar(x_vals, np.mean(a_data, axis=0), np.std(a_data, axis=0) / np.sqrt(len(a_data)),
                    fmt='o-', color='grey', label=f'A responses (n={len(a_cells)})')

    x_cells = sig_cells_ani.get('X0', [])
    if len(x_cells) > 0:
        x_data = np.mean(data_ani['activity']['gr_2'][x_cells][:, trials, poststim_frames], axis=2)
        ax.errorbar(x_vals, np.mean(x_data, axis=0), np.std(x_data, axis=0) / np.sqrt(len(x_data)),
                    fmt='o-', color='black', label=f'X responses (n={len(x_cells)})')

    ax.set_ylabel('z-ΔF/F', fontsize=13)
    ax.set_xlabel('Unexpected trial #', fontsize=13)
    ax.legend(fontsize=11, frameon=False, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax


def group_sessions_by_stimulus(table, ani_range, stim_col=3, drug_col=None):
    """{stim_label: [ani, ani, ...]}, grouping sessions by table.iloc[ani][stim_col] (e.g. 'C'/'D'/'F').

    If drug_col is given (e.g. the CNO/Saline column in the LC_DREADDs sheet), also returns:
      - drug_groups: {drug_label: [ani, ani, ...]} (e.g. 'CNO'/'Saline')
      - stim_drug_groups: {(stim_label, drug_label): [ani, ani, ...]}, e.g. the ('C', 'CNO')
        entry is the same as the hand-built 'C_CNO' list in LC_Dreadds analysis.ipynb's
        data_map, computed straight from the table instead of typed out by hand.
    In that case the return is a 3-tuple (stim_groups, drug_groups, stim_drug_groups) rather
    than the bare stim_groups dict.
    """
    stim_groups = {}
    for ani in ani_range:
        stim_groups.setdefault(table.iloc[ani][stim_col], []).append(ani)

    if drug_col is None:
        return stim_groups

    drug_groups = {}
    stim_drug_groups = {}
    for ani in ani_range:
        stim_label = table.iloc[ani][stim_col]
        drug_label = table.iloc[ani][drug_col]
        drug_groups.setdefault(drug_label, []).append(ani)
        stim_drug_groups.setdefault((stim_label, drug_label), []).append(ani)

    return stim_groups, drug_groups, stim_drug_groups


def compute_sig_cells_per_session(data, anis, **kwargs):
    """compute_sig_cells for each session. Must run before any pooling across sessions --
    significance testing needs each session's own (unpooled) trial data."""
    return {ani: compute_sig_cells(data[ani], **kwargs) for ani in anis}


def plot_qc_heatmaps_pooled(data, anis, poststim_frames, max_tri=16, figsize=(18, 6), vmin=-2, vmax=2, sort_by='self'):
    """
    Pooled version of plot_qc_heatmaps across multiple low-yield axon sessions of the same
    stimulus (see group_sessions_by_stimulus).

    Deviant ("unexpected") trials in block 1 are independently randomized within each session,
    so their absolute trial numbers don't line up across sessions -- only their ORDER does
    (e.g. "this session's 1st unexpected trial"). Each session is therefore reduced to a
    (boutons x frames) slice using its OWN unpred_trials['gr_2'] / pred-trial selection, and
    those per-session slices are concatenated across sessions along axis 0 (boutons) -- the
    same pattern this codebase already uses in plot_heatmaps/plot_unexp_psth_concat.
    """
    conditions = [
        {'activity': 'gr_1', 'kind': 'unpred', 'title': 'A (unexpected trials)'},
        {'activity': 'gr_2', 'kind': 'predicted', 'title': 'B (expected trials)'},
        {'activity': 'gr_2', 'kind': 'unpred_early', 'title': f'X (block 1, trials 1-{max_tri})'},
        {'activity': 'gr_2', 'kind': 'unpred_late', 'title': 'X (block 2)'},
    ]

    n_frames = data[anis[0]]['activity']['gr_1'].shape[2]
    condition_data = []
    for cond in conditions:
        per_session = []
        for ani in anis:
            data_ani = data[ani]
            unpred_gr2 = data_ani['unpred_trials']['gr_2']
            if cond['kind'] == 'predicted':
                unpred_gratings = iv.get_unpred_grats(data_ani)
                trials = iv.sparse_pred_trials(data_ani, unpred_gratings, max_tri, method='complex')
            elif cond['kind'] == 'unpred_early':
                trials = unpred_gr2[:max_tri]
            elif cond['kind'] == 'unpred_late':
                trials = unpred_gr2[16:]
            else:  # 'unpred' -- all of this session's unexpected trials
                trials = unpred_gr2
            if len(trials) == 0:
                continue
            per_session.append(np.mean(data_ani['activity'][cond['activity']][:, trials, :], axis=1))
        condition_data.append(np.concatenate(per_session, axis=0) if per_session else np.zeros((0, n_frames)))

    fig, axes = plt.subplots(1, len(conditions), figsize=figsize)
    for i, (cond, vals) in enumerate(zip(conditions, condition_data)):
        ax = axes[i]
        if sort_by == 'self':
            sort_idx = np.argsort(np.mean(vals[:, poststim_frames], axis=1))
        else:
            sort_idx = np.argsort(np.mean(condition_data[sort_by][:, poststim_frames], axis=1))
        sns.heatmap(vals[sort_idx], ax=ax, cmap='coolwarm', cbar=(i == len(conditions) - 1), vmin=vmin, vmax=vmax)

        ax.set_title(f"{cond['title']} (n={vals.shape[0]})", fontsize=13)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_xticks([11.5, 19, 37])
        ax.set_xticklabels(['-1', '0', '2.4'])
        ax.tick_params(axis='x', rotation=0, labelsize=11)

        if i == 0:
            ax.set_ylabel('bouton # (sorted)', fontsize=12)
            n = vals.shape[0]
            y_ticks = np.arange(0, n, max(1, n // 10))
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticks)
        else:
            ax.set_yticks([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig, axes


def plot_qc_psth_pooled(data, sig_cells, anis, groups=('gr_1', 'gr_2', 'X0'), max_tri=16, figsize=None, ylim=None):
    """
    Pooled version of plot_qc_psth: for each responsive-cell group, each session contributes
    its own (cells x frames) response -- using its own unpred_trials['gr_2'] order for A/X and
    its own sparse_pred_trials selection for B -- and these are concatenated across sessions
    along axis 0 (boutons), same rationale as plot_qc_heatmaps_pooled.
    """
    t_frames = data[anis[0]]['activity']['gr_1'].shape[2]
    figsize = figsize or (4.5 * len(groups), 4)
    fig, axes = plt.subplots(1, len(groups), figsize=figsize)
    if len(groups) == 1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        a_list, b_list, x_list = [], [], []
        for ani in anis:
            data_ani = data[ani]
            cells = sig_cells[ani].get(group, [])
            if len(cells) == 0:
                continue
            unpred_gr2 = data_ani['unpred_trials']['gr_2'][:max_tri]
            unpred_gratings = iv.get_unpred_grats(data_ani)
            pred_trials = iv.sparse_pred_trials(data_ani, unpred_gratings, max_tri, method='complex')

            a_list.append(np.mean(data_ani['activity']['gr_1'][cells][:, unpred_gr2, :], axis=1))
            b_list.append(np.mean(data_ani['activity']['gr_2'][cells][:, pred_trials, :], axis=1))
            x_list.append(np.mean(data_ani['activity']['gr_2'][cells][:, unpred_gr2, :], axis=1))

        n_cells = sum(a.shape[0] for a in a_list)
        ax.set_title(f'{group}-responsive (n={n_cells})', fontsize=13)
        if n_cells == 0:
            continue

        a_resp = np.concatenate(a_list, axis=0)
        b_resp = np.concatenate(b_list, axis=0)
        x_resp = np.concatenate(x_list, axis=0)

        viz.plot_shaded_error(ax, range(t_frames), a_resp, color='grey', style='dash', label='A')
        viz.plot_shaded_error(ax, range(t_frames), b_resp, color='steelblue', style='dot', label='B (expected)')
        viz.plot_shaded_error(ax, range(t_frames), x_resp, color='black', label='X (unexpected)')

        ax.axvline(19, linestyle='--', color='k', linewidth=1)
        ax.set_xlabel('Time from onset (s)', fontsize=12)
        ax.set_xticks([11.5, 19, 37])
        ax.set_xticklabels(['-1', '0', '2.4'])
        if ylim:
            ax.set_ylim(ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=9)

    axes[0].set_ylabel('z-ΔF/F', fontsize=12)
    plt.tight_layout()
    return fig, axes


def plot_qc_adaptation_pooled(data, sig_cells, anis, poststim_frames, figsize=(5, 4), ax=None):
    """
    Pooled version of plot_qc_adaptation. Trial POSITION (1st/2nd/... unexpected trial) is
    aligned across sessions -- not absolute trial number, since deviant timing in block 1 is
    independently randomized per session -- by slicing each session's own
    unpred_trials['gr_2'] to the shortest shared length before pooling boutons.
    """
    min_trials = min(len(data[ani]['unpred_trials']['gr_2']) for ani in anis)
    x_vals = list(range(min_trials))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    a_list = []
    for ani in anis:
        cells = sig_cells[ani].get('gr_1', [])
        if len(cells) == 0:
            continue
        trials = data[ani]['unpred_trials']['gr_2'][:min_trials]
        a_list.append(np.mean(data[ani]['activity']['gr_1'][cells][:, trials, poststim_frames], axis=2))
    if a_list:
        a_data = np.concatenate(a_list, axis=0)
        ax.errorbar(x_vals, np.mean(a_data, axis=0), np.std(a_data, axis=0) / np.sqrt(len(a_data)),
                    fmt='o-', color='grey', label=f'A responses (n={len(a_data)})')

    x_list = []
    for ani in anis:
        cells = sig_cells[ani].get('X0', [])
        if len(cells) == 0:
            continue
        trials = data[ani]['unpred_trials']['gr_2'][:min_trials]
        x_list.append(np.mean(data[ani]['activity']['gr_2'][cells][:, trials, poststim_frames], axis=2))
    if x_list:
        x_data = np.concatenate(x_list, axis=0)
        ax.errorbar(x_vals, np.mean(x_data, axis=0), np.std(x_data, axis=0) / np.sqrt(len(x_data)),
                    fmt='o-', color='black', label=f'X responses (n={len(x_data)})')

    ax.set_ylabel('z-ΔF/F', fontsize=13)
    ax.set_xlabel('Unexpected trial #', fontsize=13)
    ax.legend(fontsize=11, frameon=False, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax


def plot_axon_psth(data, anis, max_tri=16, figsize=(5, 4), ylim=None):
    """
    Single PSTH across ALL axons -- no significant-cell filtering, since axon yield per
    session is too low for that to be meaningful -- pooled across sessions of the same
    stimulus (see group_sessions_by_stimulus). Three conditions, all on the gr_2 (B/X)
    stimulus slot, matching plot_qc_heatmaps_pooled's B/X-block-1/X-block-2 conditions:
      expected B   (black) - predicted/non-deviant trials (iv.sparse_pred_trials)
      unexpected X (red)   - block 1, first max_tri deviant trials (still novel)
      expected X   (blue)  - block 2, last max_tri deviant trials (X occurs on every trial by
                             this point, so it's no longer a surprise)
    Each session contributes its own trial-reduced (boutons x frames) slice -- deviant timing
    in block 1 is independently randomized per session, so trials are always selected via each
    session's own unpred_trials['gr_2'] order, never by pooling raw activity across sessions
    first.
    """
    t_frames = data[anis[0]]['activity']['gr_2'].shape[2]
    b_list, x_unexp_list, x_exp_list = [], [], []
    for ani in anis:
        data_ani = data[ani]
        unpred_gr2 = data_ani['unpred_trials']['gr_2']
        unpred_gratings = iv.get_unpred_grats(data_ani)
        pred_trials = iv.sparse_pred_trials(data_ani, unpred_gratings, max_tri, method='complex')

        b_list.append(np.mean(data_ani['activity']['gr_2'][:, pred_trials, :], axis=1))
        x_unexp_list.append(np.mean(data_ani['activity']['gr_2'][:, unpred_gr2[:max_tri], :], axis=1))
        x_exp_list.append(np.mean(data_ani['activity']['gr_2'][:, unpred_gr2[16:16+max_tri], :], axis=1))



    b_resp = np.concatenate(b_list, axis=0)
    x_unexp_resp = np.concatenate(x_unexp_list, axis=0)
    x_exp_resp = np.concatenate(x_exp_list, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    viz.plot_shaded_error(ax, range(t_frames), b_resp, color='grey', label=f'B, expected')
    viz.plot_shaded_error(ax, range(t_frames), x_unexp_resp, color='red', label=f'X, unexpected')
    viz.plot_shaded_error(ax, range(t_frames), x_exp_resp, color='steelblue', label=f'X, expected')

    ax.axvline(19, linestyle='--', color='k', linewidth=1)
    ax.set_xlabel('Time from onset (s)', fontsize=12)
    ax.set_ylabel('z-ΔF/F', fontsize=12)
    ax.set_xticks([11.5, 19, 37])
    ax.set_xticklabels(['-1', '0', '2.4'])
    if ylim:
        ax.set_ylim(ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    ax.legend(loc='lower left', fontsize='small')

    ax.set_title(f'All axons (n={b_resp.shape[0]})', fontsize=13)

    plt.tight_layout()
    return fig, ax


def plot_axon_adaptation(data, anis, poststim_frames, figsize=(5, 4), ax=None):
    """
    Pooled adaptation curve using ALL axons -- no significant-cell filtering -- across
    sessions of the same stimulus. Trial POSITION (1st/2nd/... unexpected trial) is aligned
    across sessions -- not absolute trial number, since deviant timing in block 1 is
    independently randomized per session -- by slicing each session's own
    unpred_trials['gr_2'] to the shortest shared length before pooling boutons.
    """
    min_trials = min(len(data[ani]['unpred_trials']['gr_2']) for ani in anis)
    x_vals = list(range(min_trials))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    a_list, x_list = [], []
    for ani in anis:
        trials = data[ani]['unpred_trials']['gr_2'][:min_trials]
        a_list.append(np.mean(data[ani]['activity']['gr_1'][:, trials, poststim_frames], axis=2))
        x_list.append(np.mean(data[ani]['activity']['gr_2'][:, trials, poststim_frames], axis=2))

    a_data = np.concatenate(a_list, axis=0)
    x_data = np.concatenate(x_list, axis=0)

    ax.errorbar(x_vals, np.mean(a_data, axis=0), np.std(a_data, axis=0) / np.sqrt(len(a_data)),
                fmt='o-', color='grey', label=f'A responses (n={len(a_data)})')
    ax.errorbar(x_vals, np.mean(x_data, axis=0), np.std(x_data, axis=0) / np.sqrt(len(x_data)),
                fmt='o-', color='black', label=f'X responses (n={len(x_data)})')

    ax.set_ylabel('z-ΔF/F', fontsize=13)
    ax.set_xlabel('Unexpected trial #', fontsize=13)
    ax.legend(fontsize=11, frameon=False, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax


def run_axon_quality_check(data, anis, poststim_frames=slice(23, 33), max_tri=16, vmin=-2, vmax=2):
    """
    Axon-specific QC pass, pooled across sessions of the same stimulus (see
    group_sessions_by_stimulus): heatmaps of all axons, a single 3-condition PSTH (expected B /
    unexpected X / expected X), and an adaptation curve -- all using every axon, with no
    significant-cell filtering anywhere, since axon yield per session is too low for that to
    be meaningful.
    """
    return {
        'heatmaps': plot_qc_heatmaps_pooled(data, anis, poststim_frames, max_tri=max_tri, vmin=vmin, vmax=vmax),
        'psth': plot_axon_psth(data, anis, max_tri=max_tri),
        'adaptation': plot_axon_adaptation(data, anis, poststim_frames),
    }


def run_quality_check_pooled(data, anis, thres=0.5, prestim_frames=slice(12, 19), poststim_frames=slice(23, 33),
                              max_tri=16, method='threshold'):
    """
    The first three run_quality_check panels (heatmaps, psth, adaptation), pooled across
    multiple sessions of the same stimulus (see group_sessions_by_stimulus). Significant-cell
    detection runs per session (compute_sig_cells_per_session) before any pooling, since it
    needs each session's own trial data; the plots pool boutons across sessions using trial
    ORDER (not absolute trial number), since deviant timing in block 1 is independently
    randomized per session.
    """
    sig_cells = compute_sig_cells_per_session(data, anis, thres=thres, prestim_frames=prestim_frames,
                                               poststim_frames=poststim_frames, method=method)
    return {
        'heatmaps': plot_qc_heatmaps_pooled(data, anis, poststim_frames, max_tri=max_tri),
        'psth': plot_qc_psth_pooled(data, sig_cells, anis, max_tri=max_tri),
        'adaptation': plot_qc_adaptation_pooled(data, sig_cells, anis, poststim_frames),
    }


def _get_raw_trace(data_ani, key, aligned_key):
    """Whole-session trace, preferring the top-level key and falling back to aligned_data."""
    if data_ani.get(key) is not None:
        return np.asarray(data_ani[key], dtype=float)
    return np.asarray(data_ani['aligned_data'][aligned_key], dtype=float)


def get_behavior_traces(data_ani, lick_window=7, fps=FPS):
    """Whole-session lick rate (licks/s), median-relative pupil, and baseline-subtracted speed."""
    lick_onsets = _get_raw_trace(data_ani, 'lick', 'Lick')
    rolling_sum = pd.Series(lick_onsets).rolling(window=lick_window, min_periods=1).sum()
    lick_rate = rolling_sum.to_numpy() * fps / lick_window

    pupil_raw = _get_raw_trace(data_ani, 'pupil', 'pupil_diameter')
    pupil_med = np.nanmedian(pupil_raw)
    pupil_rel = (pupil_raw - pupil_med) / pupil_med

    speed_raw = _get_raw_trace(data_ani, 'speed', 'Speed_Absolute')
    speed_bs = speed_raw - np.nanpercentile(speed_raw, 10)

    return lick_rate, pupil_rel, speed_bs


def _pupil_rel_baseline_subtracted(data_ani, baseline_frames=slice(0, 10)):
    """Per-grating (and 'rew') pupil_rel epochs, baseline-subtracted using the first frames."""
    out = {}
    for grat, epochs in data_ani['pupil_rel'].items():
        epochs = epochs.copy().astype(float)
        bad = np.mean(np.isnan(epochs), axis=1) > 0.5
        epochs[bad] = np.nan
        baseline_mean = np.nanmean(epochs[:, baseline_frames], axis=1, keepdims=True)
        out[grat] = epochs - baseline_mean
    return out


def _fill_and_smooth(trace, sigma=0.5):
    """Linearly interpolate over never-visited (NaN) bins, then Gaussian-smooth."""
    filled = pd.Series(trace).interpolate(limit_direction='both').to_numpy()
    return gaussian_filter1d(filled, sigma=sigma)


def _position_binned_traces(data_ani, lick_rate, pupil_rel, speed_bs, bin_size=0.1):
    """Average behavior traces across corridor position, restricted to fully-predictable trials."""
    pos = np.asarray(data_ani['position'], dtype=float)
    trial_starts = data_ani['trial_start_indices']
    pred_trials = data_ani['pred_trials']

    pred_mask = np.zeros(len(pos), dtype=bool)
    for t in pred_trials:
        start = trial_starts[t]
        end = trial_starts[t + 1] if t < len(trial_starts) - 1 else len(pos)
        pred_mask[start:end] = True

    bins = np.arange(0, np.nanmax(pos) + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    n_bins = len(bins) - 1

    # bins with no samples are left as NaN (rather than 0) so a position bin that's
    # simply never visited at this resolution doesn't get plotted as "value = 0"
    lick_binned = np.full(n_bins, np.nan)
    pupil_binned = np.full(n_bins, np.nan)
    speed_binned = np.full(n_bins, np.nan)
    for b in range(n_bins):
        in_bin = (pos >= bins[b]) & (pos < bins[b + 1]) & pred_mask
        if np.any(in_bin):
            lick_binned[b] = np.nanmean(lick_rate[in_bin])
            pupil_binned[b] = np.nanmean(pupil_rel[in_bin])
            speed_binned[b] = np.nanmean(speed_bs[in_bin])

    return bin_centers, (_fill_and_smooth(lick_binned), _fill_and_smooth(pupil_binned), _fill_and_smooth(speed_binned))


def _stim_positions(data_ani):
    """Mean corridor position of each grating slot and of reward, from a fixed frame within each trial."""
    pos = np.asarray(data_ani['position'], dtype=float)
    locs = {}
    for gr, idx in data_ani['grating_indices'].items():
        idx = np.asarray(idx)
        locs[gr] = np.mean(pos[idx[:, min(20, idx.shape[1] - 1)]])
    reward_idx = np.asarray(data_ani['reward_indices'])
    locs['reward'] = np.mean(pos[reward_idx[:, min(20, reward_idx.shape[1] - 1)]])
    return locs


def plot_qc_behavior_summary(data_ani, bin_size=0.1, figsize=(11, 11), title=None, save_path=None,
                              fps=FPS, lick_window=7):
    """
    Behavioral QC summary for a session with two stimuli (gr_1 = A, gr_2 = B/X), matching the
    Summary Plot notebook's mosaic:
      A/B/C - lick rate / pupil / speed averaged across corridor position (predictable trials only)
      D     - lick-rate heatmap around reward, trial x time
      E/F/G - lick rate / pupil / speed around reward: all trials vs unexpected (X) trials
      H/I/J - lick rate / pupil / speed around gr_2 onset: all trials vs unexpected (X) trials

    fps/lick_window control the lick-rate rolling window (see get_behavior_traces) and default
    to the 7.5 fps imaging-frame-rate convention this module otherwise assumes. Override both
    when data_ani['lick'] is a raw (non-downsampled) trace at some other native sampling rate
    (e.g. quick_behavior_preprocess.build_data_ani_from_nonsampled's native_fps) -- the
    grating/reward-window panels don't need this, only the whole-session lick-rate trace does.
    """
    lick_rate, pupil_rel, speed_bs = get_behavior_traces(data_ani, lick_window=lick_window, fps=fps)
    bin_centers, (lick_smooth, pupil_smooth, speed_smooth) = _position_binned_traces(
        data_ani, lick_rate, pupil_rel, speed_bs, bin_size=bin_size)
    stim_pos = _stim_positions(data_ani)
    pupil_rel_bs = data_ani.get('pupil_rel_bs') or _pupil_rel_baseline_subtracted(data_ani)

    reward_idx = np.asarray(data_ani['reward_indices'])
    grat2_idx = np.asarray(data_ani['grating_indices']['gr_2'])
    unpred_trials = np.asarray(data_ani['unpred_trials']['gr_2'])

    mosaic = [
        ["A", "A", "A", "A", "D", "D"],
        ["B", "B", "B", "B", "D", "D"],
        ["C", "C", "C", "C", "D", "D"],
        ["E", "E", "F", "F", "G", "G"],
        ["H", "H", "I", "I", "J", "J"],
    ]
    fig, ax = plt.subplot_mosaic(mosaic, figsize=figsize, constrained_layout=True)

    # ----- A/B/C: across corridor position (predictable trials only) -----
    for key, trace, ylabel in [("A", lick_smooth, "Lick rate (lick/s)"),
                               ("B", pupil_smooth, "Pupil (relative)"),
                               ("C", speed_smooth, "Speed-baseline (cm/s)")]:
        ax[key].plot(bin_centers, trace, lw=2, color='black')
        ax[key].axvline(x=stim_pos['gr_1'], color='red', linestyle='--', linewidth=1)
        ax[key].axvline(x=stim_pos['gr_2'], color='red', linestyle='--', linewidth=1)
        ax[key].axvline(x=stim_pos['reward'], color='red', linestyle='--', linewidth=1)
        ax[key].spines['right'].set_visible(False)
        ax[key].spines['top'].set_visible(False)
        ax[key].set_ylabel(ylabel, fontsize=11)
        ax[key].tick_params(axis='both', labelsize=10)

    for label, xpos in [('A', stim_pos['gr_1']), ('B', stim_pos['gr_2']), ('R', stim_pos['reward'])]:
        ax["A"].text(xpos, 1.02, label, transform=ax["A"].get_xaxis_transform(),
                     ha='center', va='bottom', fontsize=9, color='dimgrey')

    ax["A"].set_title("Predictable trials: lick / pupil / speed across position", fontsize=12)
    ax["C"].set_xlabel("Position (a.u.)", fontsize=11)

    # ----- D: lick-rate heatmap around reward, across trials -----
    lick_rate_matrix = lick_rate[reward_idx][:, 5:35]
    vmin, vmax = np.nanpercentile(lick_rate_matrix, [2, 98])
    hm = sns.heatmap(lick_rate_matrix, ax=ax["D"], cmap='coolwarm', vmin=vmin, vmax=vmax,
                      cbar_kws={'label': 'Lick rate (lick/s)', 'shrink': 0.8})
    hm.collections[0].colorbar.ax.tick_params(labelsize=9)
    hm.collections[0].colorbar.set_label('Lick rate (lick/s)', fontsize=10)
    n_trials = lick_rate_matrix.shape[0]
    y_ticks = np.arange(0, max(1, n_trials), 50)
    ax["D"].set_yticks(y_ticks + 0.5)
    ax["D"].set_yticklabels([str(int(y)) for y in y_ticks])
    ax["D"].set_ylabel('Trial number', fontsize=11)
    ax["D"].set_xticks([0, 7.5, 15, 22.5, 30])
    ax["D"].set_xticklabels(['-2', '-1', '0', '1', '2'], rotation=0)
    ax["D"].set_xlabel('Time from reward (s)', fontsize=11)
    ax["D"].axvline(x=15, color='black', linestyle='--', linewidth=1.5)
    ax["D"].set_title('Lick rate around reward', fontsize=12)
    ax["D"].tick_params(axis='both', labelsize=10)
    ax["D"].spines['top'].set_visible(False)
    ax["D"].spines['right'].set_visible(False)

    # ----- E/F/G: near reward, all trials vs unexpected (X) trials -----
    # lick rate is a whole-session trace indexed via reward_indices; pupil/speed are
    # already trial x frame matrices (baseline-subtracted pupil, cm/s speed) keyed by 'rew'
    reward_traces = {
        "E": (lick_rate[reward_idx], 'Lick rate (lick/s)'),
        "F": (pupil_rel_bs['rew'], 'Pupil (relative)'),
        "G": (data_ani['speed_cm_s']['rew'], 'Speed-baseline (cm/s)'),
    }
    for key, (mat, ylabel) in reward_traces.items():
        t_frames = mat.shape[1]
        viz.plot_shaded_error(ax[key], np.arange(t_frames), mat,
                              color='steelblue', label='expected' if key == 'E' else None, alpha=0.2)
        viz.plot_shaded_error(ax[key], np.arange(t_frames), mat[unpred_trials, :],
                              color='black', label='unexpected' if key == 'E' else None, alpha=0.2)
        ax[key].axvline(x=20, color='red', linestyle='--', linewidth=1)
        ax[key].set_ylabel(ylabel, fontsize=11)
        ax[key].set_xlabel('Time from reward (s)', fontsize=11)
        ax[key].set_xticks([12.5, 20, 27.5])
        ax[key].set_xticklabels(['-1', '0', '1'])
        ax[key].tick_params(axis='both', labelsize=10)
        ax[key].spines['right'].set_visible(False)
        ax[key].spines['top'].set_visible(False)
    ax['E'].legend(frameon=False, fontsize=9)
    ax['E'].set_title('Around reward', fontsize=12, loc='left')

    # ----- H/I/J: near gr_2 onset, all trials vs unexpected (X) trials -----
    grat2_traces = {
        "H": (lick_rate[grat2_idx], 'Lick rate (lick/s)'),
        "I": (pupil_rel_bs['gr_2'], 'Pupil (relative)'),
        "J": (data_ani['speed_cm_s']['gr_2'], 'Speed-baseline (cm/s)'),
    }
    for key, (mat, ylabel) in grat2_traces.items():
        t_frames = mat.shape[1]
        viz.plot_shaded_error(ax[key], np.arange(t_frames), mat,
                              color='steelblue', label='expected' if key == 'H' else None, alpha=0.2)
        viz.plot_shaded_error(ax[key], np.arange(t_frames), mat[unpred_trials, :],
                              color='black', label='unexpected' if key == 'H' else None, alpha=0.2)
        ax[key].axvline(x=20, color='red', linestyle='--', linewidth=1)
        ax[key].set_ylabel(ylabel, fontsize=11)
        ax[key].set_xlabel('Time from grat 2 onset (s)', fontsize=11)
        ax[key].set_xticks([12.5, 20, 38])
        ax[key].set_xticklabels(['-1', '0', '2.4'])
        ax[key].tick_params(axis='both', labelsize=10)
        ax[key].spines['right'].set_visible(False)
        ax[key].spines['top'].set_visible(False)
    ax['H'].legend(frameon=False, fontsize=9)
    ax['H'].set_title('Around gr_2 onset', fontsize=12, loc='left')

    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold')
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax



def run_quality_check(data_ani, thres=0.5, prestim_frames=slice(12, 19), poststim_frames=slice(23, 33),
                       max_tri=16, method='threshold'):
    """
    One-shot QC pass for a single session:
      1. heatmaps of A / B(expected) / X(block 1) / X(block 2) responses
      2. PSTH of gr_1-, gr_2-, and X0-responsive neurons to A / B(expected) / X(unexpected)
      3. adaptation curves of A- and X-responsive neurons across the unexpected-trial sequence
      4. behavior (lick rate / pupil / speed) summary around the corridor, reward, and gr_2 onset
    Returns figs = {'heatmaps', 'psth', 'adaptation', 'behavior_summary'}.
    """
    sig_cells_ani = compute_sig_cells(data_ani, thres=thres, prestim_frames=prestim_frames,
                                       poststim_frames=poststim_frames, method=method)

    figs = {
        'heatmaps': plot_qc_heatmaps(data_ani, poststim_frames, max_tri=max_tri),
        'psth': plot_qc_psth(data_ani, sig_cells_ani, max_tri=max_tri),
        'adaptation': plot_qc_adaptation(data_ani, sig_cells_ani, poststim_frames, max_trials=max_tri * 2),
        'behavior_summary': plot_qc_behavior_summary(data_ani),
    }

    return sig_cells_ani, figs
