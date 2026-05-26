"""
Two-panel figure: CNO vs Saline effect on neural selectivity
Left:  Scatter — Selectivity (Saline) vs (CNO - Saline) response per cell
Right: Paired dot plot — Visual response for 3 cell groups (per animal)

Usage: paste cells into your notebook, or run as a script after loading `data`.

Assumes:
  data[sal_ani]  — Saline session (e.g. ani=5)
  data[cno_ani]  — CNO    session (e.g. ani=4)
  data[ani]['activity'][grating]  shape: (n_cells, n_trials, n_frames)
  sig_cells[ani][grating]         list of significant cell indices (from Saline)

If you have multiple matched animal pairs, set:
  cno_anis = [4, 6, 8, ...]
  sal_anis = [5, 7, 9, ...]
  (each pair cno_anis[i] <-> sal_anis[i] is the same animal)

For cross-session cell matching, load data from 'all_data_aligned' (ROIcat),
otherwise cells are assumed to share the same ROI indices across sessions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ------------------------------------------------------------------ #
#  Parameters — adjust to your experiment
# ------------------------------------------------------------------ #
cno_anis = [4]   # CNO session indices (one entry per animal)
sal_anis = [5]   # Saline session indices (same order, matched pairs)

gr_A = 'gr_1'   # grating A (predicted / familiar)
gr_B = 'gr_2'   # grating B (unexpected / novel)

prestim_frames  = slice(10, 15)
poststim_frames = slice(23, 33)

sel_threshold = 0.8   # selectivity cut-off for "highly selective to B"

# ------------------------------------------------------------------ #
#  Helper
# ------------------------------------------------------------------ #
def cell_mean_resp(data, ani, grating, cells, frames=poststim_frames):
    """Mean post-stimulus response per cell, averaged across all pred trials."""
    act = data[ani]['activity'][grating]   # (n_cells, n_trials, n_frames)
    trials = exp_tri[ani]
    return act[np.ix_(cells, trials)][:, :, frames].mean(axis=(1, 2))


# ------------------------------------------------------------------ #
#  Build per-cell arrays across all animals
# ------------------------------------------------------------------ #
sel_all      = []   # selectivity index (Saline)
delta_all    = []   # CNO - Saline response to gr_B

# For the right panel: per-animal mean response (Saline & CNO) for each group
# We will collect (sal_val, cno_val) per animal per group
group_vals = {
    'A_only':         {'sal': [], 'cno': []},   # sig for A, not B
    'B_nonselective': {'sal': [], 'cno': []},   # sig for B, |SI| < threshold
    'B_selective':    {'sal': [], 'cno': []},   # sig for B, SI > threshold
}

for sal_ani, cno_ani in zip(sal_anis, cno_anis):

    # --- significant cells (defined from Saline session) ---
    sig_A = np.array(sig_cells[sal_ani][gr_A])
    sig_B = np.array(sig_cells[sal_ani][gr_B])

    # all cells seen in either grating
    all_sig = np.union1d(sig_A, sig_B)

    # --- per-cell selectivity from Saline ---
    resp_A_sal = cell_mean_resp(data, sal_ani, gr_A, all_sig)
    resp_B_sal = cell_mean_resp(data, sal_ani, gr_B, all_sig)
    denom = np.abs(resp_A_sal) + np.abs(resp_B_sal)
    denom[denom == 0] = np.nan
    si = (resp_B_sal - resp_A_sal) / denom   # in [-1, 1]

    # --- CNO response to gr_B for the same cells ---
    resp_B_cno = cell_mean_resp(data, cno_ani, gr_B, all_sig)
    delta = resp_B_cno - resp_B_sal

    sel_all.append(si)
    delta_all.append(delta)

    # --- group masks ---
    is_sig_A = np.isin(all_sig, sig_A)
    is_sig_B = np.isin(all_sig, sig_B)

    mask_A_only  = is_sig_A & ~is_sig_B
    mask_B_non   = is_sig_B & (si <  sel_threshold)
    mask_B_sel   = is_sig_B & (si >= sel_threshold)

    # per-animal mean (gr_A for A-only group; gr_B for B groups)
    if mask_A_only.any():
        cells_A = all_sig[mask_A_only]
        group_vals['A_only']['sal'].append(cell_mean_resp(data, sal_ani, gr_A, cells_A).mean())
        group_vals['A_only']['cno'].append(cell_mean_resp(data, cno_ani, gr_A, cells_A).mean())

    if mask_B_non.any():
        cells_Bn = all_sig[mask_B_non]
        group_vals['B_nonselective']['sal'].append(cell_mean_resp(data, sal_ani, gr_B, cells_Bn).mean())
        group_vals['B_nonselective']['cno'].append(cell_mean_resp(data, cno_ani, gr_B, cells_Bn).mean())

    if mask_B_sel.any():
        cells_Bs = all_sig[mask_B_sel]
        group_vals['B_selective']['sal'].append(cell_mean_resp(data, sal_ani, gr_B, cells_Bs).mean())
        group_vals['B_selective']['cno'].append(cell_mean_resp(data, cno_ani, gr_B, cells_Bs).mean())


sel_all   = np.concatenate(sel_all)
delta_all = np.concatenate(delta_all)
n_cells   = len(sel_all)


# ------------------------------------------------------------------ #
#  Figure
# ------------------------------------------------------------------ #
fig, (ax_sc, ax_pair) = plt.subplots(1, 2, figsize=(11, 4))

# ── Left: scatter ────────────────────────────────────────────────── #
ax_sc.axhline(0, color='k', linewidth=0.5, linestyle='-')

# shade "highly selective to B" region
ax_sc.axvspan(sel_threshold, 1.05, color='lightgrey', alpha=0.6, zorder=0)

ax_sc.scatter(sel_all, delta_all,
              facecolor='none', edgecolor='black', s=18, linewidth=0.6)

ax_sc.set_xlabel('Selectivity (Saline)')
ax_sc.set_ylabel('CNO − Saline\n(z-scored ΔF/F)')
ax_sc.set_xlim([-1.05, 1.05])
ax_sc.set_xticks([-1, -0.5, 0, 0.5, 1])

ax_sc.annotate('', xy=(-0.8, 7.5), xytext=(-0.3, 7.5),
               arrowprops=dict(arrowstyle='<->', color='black', lw=1))
ax_sc.text(-1.0, 7.9, 'Selective to', fontsize=9)
ax_sc.text(-1.0, 7.3, 'A', fontsize=9)
ax_sc.text(-0.15, 7.3, 'B', fontsize=9)

ax_sc.text(-0.95, ax_sc.get_ylim()[0] + 0.3,
           f'$n$ = {n_cells}', fontsize=10, style='italic')

ax_sc.spines['top'].set_visible(False)
ax_sc.spines['right'].set_visible(False)


# ── Right: paired dot plot ───────────────────────────────────────── #
groups   = ['A_only', 'B_nonselective', 'B_selective']
x_pos    = [1, 2, 3]
labels   = ['Responsive to\nA not B', 'Non-\nselective', 'Highly\nselective']
sal_color = 'grey'
cno_color = '#E8A020'   # orange similar to the reference

for xi, grp in zip(x_pos, groups):
    sal_vals = np.array(group_vals[grp]['sal'])
    cno_vals = np.array(group_vals[grp]['cno'])
    if len(sal_vals) == 0:
        continue

    # connecting lines (one per animal)
    for sv, cv in zip(sal_vals, cno_vals):
        ax_pair.plot([xi - 0.15, xi + 0.15], [sv, cv],
                     color='grey', linewidth=0.8, alpha=0.6)

    # dots
    ax_pair.scatter([xi - 0.15] * len(sal_vals), sal_vals,
                    facecolor='none', edgecolor=sal_color, s=40, zorder=4, linewidth=1.2)
    ax_pair.scatter([xi + 0.15] * len(cno_vals), cno_vals,
                    facecolor='none', edgecolor=cno_color, s=40, zorder=4, linewidth=1.2)

    # median lines
    ax_pair.plot([xi - 0.25, xi - 0.05], [np.median(sal_vals)] * 2,
                 color='black', linewidth=2)
    ax_pair.plot([xi + 0.05, xi + 0.25], [np.median(cno_vals)] * 2,
                 color='black', linewidth=2)

    # Wilcoxon signed-rank test (or t-test if n < 5)
    if len(sal_vals) >= 5:
        _, p = stats.wilcoxon(sal_vals, cno_vals)
    else:
        _, p = stats.ttest_rel(sal_vals, cno_vals)

    y_top = max(sal_vals.max(), cno_vals.max())
    ax_pair.plot([xi - 0.15, xi + 0.15], [y_top + 0.15, y_top + 0.15], color='k', lw=0.8)
    ax_pair.text(xi, y_top + 0.2, f'P = {p:.3f}', ha='center', va='bottom', fontsize=8)

# legend
ax_pair.scatter([], [], facecolor='none', edgecolor=sal_color, s=40,
                linewidth=1.2, label='Saline')
ax_pair.scatter([], [], facecolor='none', edgecolor=cno_color, s=40,
                linewidth=1.2, label='CNO')
ax_pair.legend(frameon=False, fontsize=9, loc='upper left')

ax_pair.set_xticks(x_pos)
ax_pair.set_xticklabels(labels, ha='center')
ax_pair.set_ylabel('Visual stimulus response\n(z-scored ΔF/F)')
ax_pair.set_xlim([0.5, 3.5])

# underline "Responsive to B" groups
ax_pair.annotate('', xy=(2.35, -0.15), xytext=(3.35, -0.15),
                 xycoords=('data', 'axes fraction'),
                 arrowprops=dict(arrowstyle='-', color='black', lw=1.2))
ax_pair.text(2.85, -0.22, 'Responsive to B',
             ha='center', va='top', transform=ax_pair.get_xaxis_transform(),
             fontsize=9)

ax_pair.spines['top'].set_visible(False)
ax_pair.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('selectivity_cno_vs_saline.pdf', bbox_inches='tight')
plt.show()
