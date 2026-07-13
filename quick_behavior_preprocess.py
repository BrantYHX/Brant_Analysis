"""
Runs just the behavior half of preprocess_behav2p.DataProcessor.preprocessing()'s bonsai
branch, stopping right after DataLoader.align_events -- skips Suite2pLoader.process_suite2p_data
(the expensive F/Fneu/dF-F/z-score computation) and DataLoader.align_data (which needs that
dff data), since neither is needed just to look at the behavior data.

Does not modify preprocess_behav2p.py. As a side effect (same as the full pipeline),
align_events writes both behavior_data_nonsampled.pkl (raw, full-resolution) and data.pkl
(downsampled to one row per imaging volume) into that session's behavior_data_path.
"""
import os
import numpy as np
import pandas as pd
import preprocess_behav2p as preprocess
import init_vars as iv

IMAGING_FPS_CONVENTION = 7.5  # the frame rate this codebase's grating/reward window widths assume


def get_suite2p_metadata(suite2ppath):
    """
    n_planes, plus the two values align_events' pupil-based scanner-time fallback needs
    (n_frames_expected, fs_expected) -- all straight from suite2p's ops.npy metadata
    (ops['nplanes'], ops['nframes'], ops['fs']; see detect_scanner_time_from_pupil's
    docstring), without loading F.npy/Fneu.npy or doing any dF/F/z-score computation.
    """
    ops = np.load(os.path.join(suite2ppath, 'plane0', 'ops.npy'), allow_pickle=True).item()
    return ops['nplanes'], ops.get('nframes'), ops.get('fs')


PUPIL_COLS = {'pupil_area', 'pupil_diameter', 'pupil_long_axis', 'pupil_center_x', 'pupil_center_y'}


def add_raw_pupil_to_nonsampled(data_loader, digital_data, fs_digital, behavior_data_path, verbose=True):
    """
    align_events only merges pupil metrics into the DOWNSAMPLED dataframe, never the raw
    merged_data it saves as behavior_data_nonsampled.pkl -- so this fills that gap without
    touching preprocess_behav2p.py: re-loads that pickle, computes pupil DIAMETER directly
    from the DLC output at full (un-binned) resolution, aligns it to the file's own 'Time'
    column by nearest pupil-camera frame (the same searchsorted+clip pattern align_events
    uses for Piezo_Signal), adds it as a 'pupil' column, and re-saves.
    """
    path = os.path.join(behavior_data_path, "behavior_data_nonsampled.pkl")
    merged_data = pd.read_pickle(path)
    if 'pupil' in merged_data.columns:
        return merged_data

    dlc_path = data_loader.find_pupil_dlc_output(behavior_data_path)
    if dlc_path is None:
        if verbose:
            print(f"no DLC pupil file found in {behavior_data_path} -- 'pupil' column not added "
                  f"to behavior_data_nonsampled.pkl.")
        return merged_data

    camera_signal = data_loader.extract_digital_channel(digital_data, 6)  # P0.6, same channel align_events uses
    camera_time = data_loader.detect_falling_edges(camera_signal) / fs_digital

    x, y, p = data_loader.load_dlc_points(dlc_path)
    pupil_metrics_raw = data_loader.extract_pupil_from_dlc(x, y, p, jump_thresh=20, min_points=5, min_coverage=0.5)

    n_cam = min(len(camera_time), x.shape[0])
    cam = np.asarray(camera_time)[:n_cam]
    cam_idx = np.searchsorted(cam, merged_data['Time'].values)
    cam_idx = np.clip(cam_idx, 0, n_cam - 1)
    merged_data['pupil'] = pupil_metrics_raw['pupil_diameter'][:n_cam][cam_idx]

    merged_data.to_pickle(path)
    if verbose:
        n_valid = merged_data['pupil'].notna().sum()
        print(f"added 'pupil' column to {path}: {n_valid}/{len(merged_data)} rows have a valid value.")
    return merged_data


def load_downsampled_behavior(ani, table, sheet_name, verbose=True):
    """
    Behavior-only preprocessing for one session. Returns the downsampled (one row per
    imaging volume) behavior dataframe -- the same object saved as data.pkl by the full
    pipeline. Only supports bonsai-type sessions (table.iloc[ani][1] != 1); behav2p sessions
    load behavior straight out of the matlab file and never go through align_events.
    """
    if table.iloc[ani][1] == 1:
        raise ValueError(f"ani={ani} is a behav2p-type session (table.iloc[{ani}][1] == 1) -- "
                          f"its behavior data comes from load_matlab_variables, not align_events, "
                          f"so this shortcut doesn't apply.")

    suite2ppath = table.iloc[ani][5]
    behavior_data_path = table.iloc[ani][2]

    data_loader = preprocess.DataLoader(behavior_data_path, sheet_name, suite2ppath)
    n_planes, n_frames_expected, fs_expected = get_suite2p_metadata(suite2ppath)

    if verbose:
        dlc_path = data_loader.find_pupil_dlc_output(behavior_data_path)
        if dlc_path is None:
            print(f"ani={ani}: no DLC pupil-tracking file found in {behavior_data_path} "
                  f"(expected a name matching *PupilCamera*DLC*.csv/.h5) -- pupil columns will be absent.")
        else:
            print(f"ani={ani}: found DLC pupil file {dlc_path}")

    analog_time, analog_data, fs_analog = data_loader.load_analog_data()
    digital_time, digital_data, fs_digital = data_loader.load_digital_data()
    encoder_csv_data, vr_csv_data, events_csv_data, counter_csv_data, quadsync_csv_data, log_file, pupil_data = \
        data_loader.load_csv_data()

    downsampled, all_pupil_data, piezo_full, time_full, trial_start_indices_full, reward_indices_full, \
        scanner_time_is_reconstructed, scanner_time_diagnostics = data_loader.align_events(
            digital_data, counter_csv_data, events_csv_data, quadsync_csv_data, vr_csv_data,
            encoder_csv_data, pupil_data, analog_data, fs_analog, fs_digital, n_planes,
            n_frames_expected=n_frames_expected, fs_expected=fs_expected)

    if scanner_time_is_reconstructed:
        print(f"NOTE: ani={ani} used pupil-based scanner_time reconstruction "
              f"(scanner DAQ channel was missing). Diagnostics: {scanner_time_diagnostics}")

    add_raw_pupil_to_nonsampled(data_loader, digital_data, fs_digital, behavior_data_path, verbose=verbose)

    if verbose:
        present = PUPIL_COLS.intersection(downsampled.columns)
        if not present:
            if scanner_time_is_reconstructed:
                reason = ("scanner time was reconstructed FROM pupil-tracking confidence dropout -- "
                          "that dropout is exactly the imaging epoch where the 2P laser blocks the "
                          "camera's view of the pupil, so there may be no recoverable pupil signal "
                          "for most/all of this session, not a bug")
            elif dlc_path is None:
                reason = "no DLC file was found for this session (see message above)"
            else:
                reason = ("a DLC file was found but pupil extraction inside align_events raised an "
                          "exception it catches and only prints (look for a 'WARNING: Failed to "
                          "compute pupil metrics' line above)")
            print(f"ani={ani}: no pupil columns in the result -- {reason}.")
        else:
            n_valid = downsampled[sorted(present)[0]].notna().sum()
            print(f"ani={ani}: pupil columns present {sorted(present)}, "
                  f"{n_valid}/{len(downsampled)} rows have a valid value.")

    return downsampled


def load_downsampled_behavior_many(ani_range, table, sheet_name):
    """{ani: downsampled_behavior_df} for every ani in ani_range, skipping (with a printed
    warning) any behav2p-type sessions this shortcut doesn't apply to."""
    out = {}
    for ani in ani_range:
        if table.iloc[ani][1] == 1:
            print(f"skipping ani={ani}: behav2p-type session, not supported by this shortcut.")
            continue
        print(f"ani={ani}: loading behavior data...")
        out[ani] = load_downsampled_behavior(ani, table, sheet_name)
    return out


def get_native_fps(behavior_df):
    """Empirical sampling rate of a raw (non-downsampled) behavior dataframe, from its own
    'Time' column -- don't assume/hardcode a rate, this is a raw bonsai/VR log, not an
    imaging-frame-rate dataframe, and its native rate isn't fixed across rigs/sessions."""
    dt = np.nanmedian(np.diff(behavior_df['Time'].values))
    return 1.0 / dt


def align_trials_from_raw_behavior(behavior_df, expected_stim_set=(2, 4)):
    """
    Grating/reward trial windowing + pred/unpred classification, computed directly from a
    raw (non-downsampled) behavior dataframe. Same core logic as
    preprocess_behav2p.DataLoader.align_data (trial/grating detection, n_gratings inference,
    the {2, 4} = "always expected" stim-code convention already used to build this lab's
    existing all_data.pkl for bonsai-type sessions), MINUS align_frames (which exists only to
    trim behavior to match an imaging dff array's frame count -- irrelevant with no imaging
    data at all), and with one fix:

    align_data's reward window uses dt = np.nanmedian(np.diff(time_values)) -- derived from
    the INPUT dataframe's own resolution. On downsampled (~7.5 fps) data that happens to give
    the intended ~5.3s-wide reward window; reused directly on a raw, natively-sampled
    dataframe, the same fixed pre_frames=20/post_frames=20 would collapse to a tiny fraction
    of a real second instead. Here the reward dt is hardcoded to 1/7.5s, matching the same
    real-time span this codebase's reward-window conventions (xticks, onset-column-20, etc.)
    already assume, regardless of the source dataframe's actual sampling rate. The grating
    window's dt (7.9/60) was already resolution-independent in the original and is unchanged.
    """
    aligned_data = behavior_df.reset_index(drop=True)
    gratings_start = aligned_data[~aligned_data['Stim'].isin([0, np.nan])].index
    gratings_start = gratings_start[np.r_[True, np.diff(gratings_start) > 2]]
    trial_start_indices = aligned_data[aligned_data['Teleport'] == 1].index.tolist()
    condition = (gratings_start > trial_start_indices[0]) & (gratings_start < trial_start_indices[-1])
    gratings_start = gratings_start[condition]

    reward_bins = np.asarray(aligned_data.index[aligned_data['Reward'] == 1])
    trial_starts = np.asarray(trial_start_indices)
    trial_ends = np.r_[trial_starts[1:], len(aligned_data)]
    reward_delivery = []
    for a, b in zip(trial_starts, trial_ends):
        r = reward_bins[(reward_bins >= a) & (reward_bins < b)]
        reward_delivery.append(int(r[0]) if len(r) else None)
    reward_delivery = [r for r in reward_delivery if r is not None]

    n_trials = max(len(trial_start_indices) - 1, 1)
    n_gratings = round(len(gratings_start) / n_trials)
    grating_onsets_dict = {f'gr_{i+1}': [] for i in range(n_gratings)}
    for i, idx in enumerate(gratings_start):
        grating_num = (i % n_gratings) + 1
        grating_onsets_dict[f'gr_{grating_num}'].append(idx)

    time_values = aligned_data['Time'].values

    # gratings: fixed 60-sample window spanning ~7.9s -- dt is already hardcoded (resolution-
    # independent) in the original align_data, so this part is unchanged.
    pre_frames, post_frames = 20, 40
    total_frames = pre_frames + post_frames
    dt_gr = 7.9 / total_frames
    grating_indices = {}
    for grating, onset_indices in grating_onsets_dict.items():
        expanded = []
        for onset_idx in onset_indices:
            onset_time = aligned_data['Time'].iloc[onset_idx]
            target_times = onset_time + (np.arange(total_frames) - pre_frames) * dt_gr
            idxs = np.searchsorted(time_values, target_times)
            idxs = np.clip(idxs, 0, len(time_values) - 1)
            expanded.append(idxs)
        grating_indices[grating] = expanded

    # reward: fixed 40-sample window -- dt hardcoded to 1/7.5s (see docstring) instead of
    # np.nanmedian(np.diff(time_values)), so the real-time span stays ~5.3s regardless of
    # this dataframe's native sampling rate.
    post_frames = pre_frames
    total_frames = pre_frames + post_frames
    dt_rew = 1.0 / IMAGING_FPS_CONVENTION
    reward_indices = []
    for ridx in reward_delivery:
        reward_time = aligned_data['Time'].iloc[ridx]
        target_times = reward_time + (np.arange(total_frames) - pre_frames) * dt_rew
        idxs = np.searchsorted(time_values, target_times)
        idxs = np.clip(idxs, 0, len(time_values) - 1)
        reward_indices.append(idxs)

    unpred_trials = {f'gr_{i+1}': [] for i in range(n_gratings)}
    all_stimuli = set(np.unique(aligned_data['Stim'].dropna()))
    unpred_stim_set = all_stimuli - set(expected_stim_set)
    pred_trials = []
    for gr, indices in grating_onsets_dict.items():
        for i, idx in enumerate(indices):
            if aligned_data.loc[idx, 'Stim'] in unpred_stim_set:
                unpred_trials[gr].append(i)
            else:
                pred_trials.append(i)

    return {
        'n_gratings': n_gratings,
        'grating_indices': grating_indices,
        'reward_indices': reward_indices,
        'trial_start_indices': trial_start_indices,
        'pred_trials': pred_trials,
        'unpred_trials': unpred_trials,
    }


def build_data_ani_from_nonsampled(behavior_df, expected_stim_set=(2, 4), pupil_baseline_frames=slice(12, 19)):
    """
    Builds a data_ani dict straight from a raw (non-downsampled) behavior dataframe --
    everything Quality_Check.plot_qc_behavior_summary needs (position, pupil, speed, lick,
    pupil_rel_bs, speed_cm_s, grating_indices, reward_indices, pred_trials, unpred_trials,
    trial_start_indices) -- with NO neural/activity data, so only plot_qc_behavior_summary
    (not run_quality_check, which also needs data_ani['activity']) applies to the result.

    Returns (data_ani, native_fps). native_fps is the dataframe's own empirically-measured
    sampling rate (see get_native_fps) -- pass it to plot_qc_behavior_summary's fps= (and a
    matching lick_window=round(native_fps) for a ~1s rolling window), since it will NOT
    generally be the 7.5 fps imaging-frame-rate convention the rest of this codebase assumes.
    """
    behavior_df = behavior_df.reset_index(drop=True)
    native_fps = get_native_fps(behavior_df)
    trial_info = align_trials_from_raw_behavior(behavior_df, expected_stim_set=expected_stim_set)

    data_ani = {
        'position': behavior_df['Averaged_Position'],
        'pupil': behavior_df['pupil'],
        'speed': behavior_df['Speed_Absolute'],
        'lick': behavior_df['Lick'],
        **trial_info,
    }

    data_ani = iv.process_pupil(data_ani, baseline_frames=pupil_baseline_frames.stop or 5)
    data_ani = iv.process_speed(data_ani)

    data_ani['pupil_rel_bs'] = {}
    for grat, epochs in data_ani['pupil_rel'].items():
        epochs = epochs.copy().astype(float)
        bad = np.mean(np.isnan(epochs), axis=1) > 0.5
        epochs[bad] = np.nan
        baseline_mean = np.nanmean(epochs[:, pupil_baseline_frames], axis=1, keepdims=True)
        data_ani['pupil_rel_bs'][grat] = epochs - baseline_mean

    return data_ani, native_fps
