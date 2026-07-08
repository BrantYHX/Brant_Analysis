# MODULES
import os
from os import path, replace
import yaml
from pathlib import Path
from typing import Optional
import importlib.util
import numpy as np
import pandas as pd
import scipy.io
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from sklearn.linear_model import LinearRegression
from skimage.measure import EllipseModel
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, behavior_data_path, sheet_name, suite2ppath):
        self.behavior_data_path = behavior_data_path
        self.sheet_name = sheet_name
        self.suite2ppath = suite2ppath

    @staticmethod
    def extract_trial_type(filename,label):
        with open(filename, 'r') as f:
            yaml_string = f.read()
        trial_strings = yaml_string.strip().split('\n\n') 
        trial_type_labels = []
        for trial_string in trial_strings:
            trial = yaml.safe_load(trial_string)
            if label in trial:
                trial_type_labels.append(trial[label])
        return trial_type_labels
    
    def load_matlab_variables(self, mat_data):
        n_gratings = mat_data['n_gratings']
        def struct_to_dict(mat_struct):
            return {key: getattr(mat_struct, key, []) for key in dir(mat_struct) if not key.startswith('_')}

        def to_list(mat_obj):
            """Convert mat_struct or numpy array to list, handling both loading formats."""
            if hasattr(mat_obj, '_fieldnames'):  # bare mat_struct
                return {field: np.array(getattr(mat_obj, field)).squeeze().tolist()
                        for field in mat_obj._fieldnames}
            else:  # numpy array wrapping a mat_struct
                return mat_obj.squeeze().tolist()

        unpred_trials = to_list(mat_data['unpred_trials'])
        pred_trials = mat_data['pred_trials'].squeeze().tolist()
        trial_start_indices = mat_data['trial_start_indices'].squeeze().tolist()
        grating_indices = struct_to_dict(mat_data['grating_indices'])
        for gr in range(1, n_gratings + 1):
            key = f'gr_{gr}'
            if key in grating_indices:
                grating_indices[key] = grating_indices[key].squeeze().tolist()
            else:
                grating_indices[key] = []  

        position = mat_data['position'].squeeze()
        bin_width = 0.25
        position_flat = position.flatten()
        num_bins = int((np.max(position_flat) - np.min(position_flat)) / bin_width) + 1
        position_tunnel = pd.cut(position_flat, bins=np.arange(0, max(position_flat) + bin_width, bin_width))
        position_tunnel = pd.Categorical(position_tunnel)
        n_trials = len(grating_indices.get(f'gr_{n_gratings}', []))

        unpred_led = mat_data.get('unpred_led', np.array([])).squeeze().tolist()
        unpred_noled = mat_data.get('unpred_noled', np.array([])).squeeze().tolist()
        catchA = mat_data.get('catchA', np.array([])).squeeze().tolist()
        catchB = mat_data.get('catchB', np.array([])).squeeze().tolist()

        if 'speed' not in mat_data:
            speed = []
        else:
            speed = mat_data['speed'].squeeze()
        if 'lick' not in mat_data:
            lick = []
        else:
            lick = mat_data['lick'].squeeze()

        if 'pupil' not in mat_data:
            pupil = []
        elif np.isscalar(mat_data['pupil']) and np.isnan(mat_data['pupil']):
            pupil = []
        else:
            pupil = mat_data['pupil'].squeeze()

        if 'reward_indices' not in mat_data:
            reward_indices = []
        else:
            reward_indices = mat_data['reward_indices'].squeeze()

        return n_gratings,unpred_trials, pred_trials, trial_start_indices, grating_indices, reward_indices, position, num_bins, position_tunnel, n_trials, unpred_led, unpred_noled, catchA, catchB, speed, lick, pupil

    @staticmethod    
    def extract_digital_channel(digital_data, channel_number):
        #channel_number goes from 0 to 7
        return digital_data & (1 << channel_number) > 0
    
    @staticmethod
    def detect_edges(signal):
        edges = np.where(np.abs(np.diff(signal.astype(int))) == 1)[0] + 1
        return edges

    @staticmethod
    def detect_falling_edges(signal):
        edges = np.where((np.array(signal[:-1]) == True) & (np.array(signal[1:]) == False))[0] + 1
        return edges

    def load_excel_data(self):
        return pd.read_excel(self.behavior_data_path, sheet_name=self.sheet_name)
    
    def load_matlab_data(self):
        return scipy.io.loadmat(os.path.join(self.suite2ppath, 'data.mat'), struct_as_record=False, squeeze_me=True)
    
    def load_analog_data(self, fs_analog=1000):
        analog_file = os.path.join(self.behavior_data_path, 'AnalogInput.bin')
        raw = np.fromfile(analog_file, dtype=np.float64)
        # analog_data = np.fromfile(analog_file, dtype=np.float32).reshape(-1, 2).transpose()
        n_ch = 3 if raw.size % 3 == 0 else 2
        analog_data = raw.reshape(-1, n_ch).T
        analog_time = np.arange(analog_data[0, :].shape[0]) / fs_analog
        return analog_time, analog_data, fs_analog

    def load_digital_data(self, fs_digital=3000):
        digital_file = os.path.join(self.behavior_data_path, 'DigitalInput.bin')
        digital_data = np.fromfile(digital_file, dtype=np.byte)
        digital_time = np.arange(digital_data.shape[0]) / fs_digital
        return digital_time, digital_data, fs_digital

    def load_csv_data(self):
        encoder_csv_data = pd.read_csv(os.path.join(self.behavior_data_path, 'RotaryEncoder.csv'))
        vr_csv_data = pd.read_csv(os.path.join(self.behavior_data_path, 'VrPosition.csv'))
        events_csv_data = pd.read_csv(os.path.join(self.behavior_data_path, 'FrameEvents.csv'))
        counter_csv_data = pd.read_csv(os.path.join(self.behavior_data_path, 'FrameCounter.csv'))
        quadsync_csv_data = pd.read_csv(os.path.join(self.behavior_data_path, 'QuadSynch.csv'))
        log_file = os.path.join(self.behavior_data_path,'TrialLogging.yaml')
        pupil_data = pd.read_csv(os.path.join(self.behavior_data_path, 'PupilCamera_CameraMetadata.csv'))
        
        return encoder_csv_data, vr_csv_data, events_csv_data, counter_csv_data, quadsync_csv_data, log_file, pupil_data

    def clean_oscillating_signal(self, photodiode_signal, fs, window_size_ms=15):
        window_samples = int(window_size_ms * fs / 1000)
        if window_samples % 2 == 0:
            window_samples += 1
            
        signal_float = photodiode_signal.astype(float)
        cleaned_signal = signal.medfilt(signal_float, kernel_size=window_samples)
        cleaned_signal = cleaned_signal > 0.5
        return cleaned_signal
    
    @staticmethod
    def find_pupil_dlc_output(behavior_data_path: str) -> Optional[str]:
        bp = Path(behavior_data_path)

        # Prefer CSV always
        cands = (
            sorted(bp.glob("*PupilCamera*DLC*filtered*.csv")) +
            sorted(bp.glob("*PupilCamera*DLC*.csv"))
        )
        if cands:
            return str(cands[0])

        has_tables = importlib.util.find_spec("tables") is not None
        if not has_tables:
            return None

        cands = (
            sorted(bp.glob("*PupilCamera*DLC*filtered*.h5")) +
            sorted(bp.glob("*PupilCamera*DLC*.h5"))
        )
        return str(cands[0]) if cands else None

    @staticmethod
    def load_dlc_points(dlc_path: str):
        """Return x,y,p arrays (frames x points). Works for DLC CSV or H5.
        Note: H5 needs pytables; if missing, user should install or export CSV.
        """
        dlc_path = Path(dlc_path)
        if dlc_path.suffix.lower() == ".h5":
            df = pd.read_hdf(dlc_path)  # requires pytables
            x = df.xs("x", level="coords", axis=1).to_numpy()
            y = df.xs("y", level="coords", axis=1).to_numpy()
            p = df.xs("likelihood", level="coords", axis=1).to_numpy()

        else:
            df = pd.read_csv(dlc_path, skiprows=3, header=None)
            # DLC csv is [x1 y1 p1 x2 y2 p2 ...] (often with a first "frame" col)
            arr = df.to_numpy()
            if arr.shape[1] % 3 == 1:
                arr = arr[:, 1:]
            x = arr[:, 0::3]
            y = arr[:, 1::3]
            p = arr[:, 2::3]

        conf_mask = p >= 0.97
        x = np.where(conf_mask, x, np.nan)
        y = np.where(conf_mask, y, np.nan)

        return x, y, p

    @staticmethod
    def extract_pupil_from_dlc(x, y, p, jump_thresh=20, min_points=5,min_coverage=0.5):
        n = x.shape[0]

        good = p >= min_coverage 
        dx = np.abs(np.diff(x, axis=0, prepend=x[[0]]))
        dy = np.abs(np.diff(y, axis=0, prepend=y[[0]]))
        good &= ~((dx > jump_thresh) | (dy > jump_thresh))
        
        area = np.full(n, np.nan)
        diameter = np.full(n, np.nan)
        long_axis = np.full(n, np.nan)
        cx = np.full(n, np.nan)
        cy = np.full(n, np.nan)

        ell = EllipseModel()

        for i in range(n):
            ok = good[i] & np.isfinite(x[i]) & np.isfinite(y[i])
            xi, yi = x[i, ok], y[i, ok]
            if xi.size < min_points:
                continue

            mx, my = np.median(xi), np.median(yi)
            madx = np.median(np.abs(xi - mx)) + 1e-9
            mady = np.median(np.abs(yi - my)) + 1e-9
            keep = (np.abs(xi - mx) < 3 * madx) & (np.abs(yi - my) < 3 * mady)  # change to 3 so that i dont drop below threshold
            xi, yi = xi[keep], yi[keep]
            if xi.size < min_points:
                continue

            if not ell.estimate(np.column_stack([xi, yi])):
                continue

            xc, yc, a, b, _ = ell.params
            area[i] = np.pi * a * b
            long_axis[i] = max(a, b)
            diameter[i] = 2 * np.sqrt(a * b)
            cx[i], cy[i] = xc, yc

        return {
            "pupil_area": area,
            "pupil_diameter": diameter,
            "pupil_long_axis": long_axis,
            "pupil_center_x": cx,
            "pupil_center_y": cy,
        }

    def estimate_offset(self, gpu_times, diode_times, max_offset=5.0, bin_size=0.01):
        # Convert to binary time series
        t_min = 0
        t_max = max(gpu_times.max(), diode_times.max()) + max_offset
        bins = np.arange(t_min, t_max, bin_size)

        gpu_hist, _ = np.histogram(gpu_times, bins)
        diode_hist, _ = np.histogram(diode_times, bins)
        corr = correlate(diode_hist, gpu_hist, mode='full')
        lags = np.arange(-len(gpu_hist)+1, len(gpu_hist)) * bin_size
        best_offset = lags[np.argmax(corr)]
        return best_offset

    def detect_scanner_time_from_pupil(self, camera_time, dlc_path, n_volumes_expected, fs_per_plane_expected, n_planes, min_run_frames=10, max_gap_frames=60):
        """
        Fallback for sessions where the scanner digital channel (P0.0) wasn't
        recorded (e.g. after a DAQ power surge). Reconstructs per-frame
        scanner timestamps using the fact that the pupil camera can't see the
        pupil while the 2p scanner/laser is on -- so DLC tracking confidence
        drops sharply for the duration of imaging, bracketing the imaging
        epoch in the camera's (already-synced) time base. Within that
        bracketed window, per-frame times are filled in via constant-rate
        interpolation using the known frame count and frame rate from
        Suite2p/ScanImage metadata (ops['nframes'], ops['fs']), since there
        is no remaining signal that captures true scanner jitter.
        """

        x, y, p = self.load_dlc_points(dlc_path)

        mean_p = np.nanmean(p, axis=1)
        imaging_mask = mean_p >= 0.97

        T = min(len(imaging_mask), len(camera_time))
        imaging_mask = imaging_mask[:T]
        cam_t = camera_time[:T]

        diff = np.diff(imaging_mask.astype(int))
        run_starts = np.where(diff == 1)[0] + 1
        run_ends = np.where(diff == -1)[0] + 1
        if imaging_mask[0]:
            run_starts = np.r_[0, run_starts]
        if imaging_mask[-1]:
            run_ends = np.r_[run_ends, len(imaging_mask)]

        run_lengths_raw = run_ends - run_starts

        if len(run_starts) > 1:
            merged_starts = [run_starts[0]]
            merged_ends = [run_ends[0]]
            for s, e in zip(run_starts[1:], run_ends[1:]):
                gap = s - merged_ends[-1]
                if gap <= max_gap_frames:
                    merged_ends[-1] = e  # bridge: extend current run
                else:
                    merged_starts.append(s)
                    merged_ends.append(e)
            run_starts = np.array(merged_starts)
            run_ends = np.array(merged_ends)

        run_lengths = run_ends - run_starts
        valid = run_lengths >= min_run_frames

        diagnostics = {
            "n_candidate_runs_raw": len(run_lengths_raw),
            "run_lengths_raw": run_lengths_raw.tolist(),
            "n_candidate_runs_after_merge": len(run_starts),
            "n_valid_runs": int(valid.sum()),
            "run_lengths": run_lengths.tolist(),
            "max_gap_frames_used": max_gap_frames,
        }

        longest_idx = np.argmax(np.where(valid, run_lengths, -1))
        start_idx, end_idx = run_starts[longest_idx], run_ends[longest_idx] - 1

        imaging_start_time = cam_t[start_idx]
        imaging_end_time = cam_t[end_idx]

        diagnostics["chosen_run_length_frames"] = int(run_lengths[longest_idx])
        diagnostics["start_camera_frame_idx"] = int(start_idx)
        diagnostics["end_camera_frame_idx"] = int(end_idx)

        epoch_duration = imaging_end_time - imaging_start_time
        expected_duration = n_volumes_expected / fs_per_plane_expected
        diagnostics["epoch_duration_from_pupil_s"] = epoch_duration
        diagnostics["expected_duration_from_metadata_s"] = expected_duration
        diagnostics["duration_discrepancy_s"] = epoch_duration - expected_duration
        diagnostics["duration_discrepancy_pct"] = 100 * (epoch_duration - expected_duration) / expected_duration

        # Build one timestamp PER SINGLE-PLANE FRAME (n_volumes_expected *
        # n_planes total), at the single-plane sampling rate
        # (fs_per_plane_expected * n_planes) -- this is the array shape
        # decimate_dataframe expects, matching what detect_falling_edges on
        # the real scanner digital channel would have produced (one falling
        # edge per plane acquisition, n_planes of them per volume).
        n_single_plane_frames = n_volumes_expected * n_planes
        fs_single_plane = fs_per_plane_expected * n_planes
        dt = 1.0 / fs_single_plane
        scanner_time = imaging_start_time + dt * np.arange(n_single_plane_frames)
        diagnostics["n_single_plane_frames_generated"] = n_single_plane_frames
        diagnostics["scanner_time_end_vs_pupil_end_diff_s"] = scanner_time[-1] - imaging_end_time

        return scanner_time, diagnostics

    def align_events(self, digital_data, counter_csv_data, events_csv_data, quadsync_csv_data, vr_csv_data, encoder_csv_data, pupil_data,analog_data, fs_analog, fs_digital, n_planes, n_frames_expected=None, fs_expected=None):
        # Align synchronization events between the GPU and the photodiode signal
        min_nidaq_sample = 10_000
        photodiode_signal_raw = self.extract_digital_channel(digital_data, 2)
        cleaned_signals = {}
        photodiode_signal_clean = self.clean_oscillating_signal(photodiode_signal_raw, fs_digital,window_size_ms=5)
        gpu_toggle_time = quadsync_csv_data["Time"].values
        diode_time = self.detect_edges(photodiode_signal_clean) / fs_digital
        # extract scanner times
        scanner_signal = self.extract_digital_channel(digital_data, 0)
        scanner_time = self.detect_falling_edges(scanner_signal) / fs_digital # the end of a scanning cycle or acquisition frame.
        camera_signal = self.extract_digital_channel(digital_data, 6)  # P0.6
        camera_edges = self.detect_falling_edges(camera_signal) / fs_digital
        camera_time = np.asarray(camera_edges)  

        # --- Fallback: scanner channel missing/empty (DAQ power surge) ---
        scanner_time_diagnostics = None
        scanner_time_is_reconstructed = False
        if len(scanner_time) == 0:
            print("WARNING: No scanner signal detected on P0.0 -- falling back to "
                  "pupil-visibility-based scanner time reconstruction.")
            dlc_path = self.find_pupil_dlc_output(self.behavior_data_path)
            if dlc_path is None:
                raise ValueError(
                    "Scanner signal missing AND no DLC pupil output found -- "
                    "cannot reconstruct scanner timing for this session. Manual "
                    "intervention needed."
                )
            # NOTE: detect_scanner_time_from_pupil returns ONE TIMESTAMP PER
            # SINGLE-PLANE FRAME (length n_frames_expected * n_planes), not
            # per volume, since that's what decimate_dataframe expects (it
            # downsamples by n_planes itself). Passing n_planes here 
            scanner_time, scanner_time_diagnostics = self.detect_scanner_time_from_pupil(
                camera_time, dlc_path, n_frames_expected, fs_expected, n_planes
            )
            scanner_time_is_reconstructed = True

        # fig, axes = plt.subplots(8, 1, figsize=(15, 10), sharex=True)
        # t = np.arange(len(digital_data)) / fs_digital

        # for ch in range(8):
        #     sig = self.extract_digital_channel(digital_data, ch)
        #     axes[ch].plot(t, sig)
        #     axes[ch].set_ylabel(f'P0.{ch}')
        #     axes[ch].set_ylim(-0.1, 1.1)
        #     axes[ch].set_xlim([5, 15])


        # axes[-1].set_xlabel('Time (s)')
        
        # plt.tight_layout()
        # plt.show()
        try:
            first_diode_time = diode_time[diode_time > (min_nidaq_sample / fs_digital)][0]
        except IndexError:
            print("WARNING: No photodiode detected")
            intercept_avg = 1.511616
            slope_avg = 1.000
            predicted_photodiode_time = gpu_toggle_time * slope_avg + intercept_avg
            diode_time = predicted_photodiode_time  # replace missing photodiode data
            first_diode_time = diode_time[diode_time > (min_nidaq_sample / fs_digital)][0]

        gpu_v_photodiode = np.empty((gpu_toggle_time.shape[0], 2))
        for synch_idx, gpu_t in enumerate(gpu_toggle_time):
            if synch_idx == 0:
                gpu_v_photodiode[synch_idx, 0] = gpu_t
                gpu_v_photodiode[synch_idx, 1] = first_diode_time
            else:
                expected_delta = gpu_t - gpu_v_photodiode[synch_idx - 1, 0]
                expected_diode_time = gpu_v_photodiode[synch_idx - 1, 1] + expected_delta
                argmin = np.argmin(abs(diode_time - expected_diode_time))
                gpu_v_photodiode[synch_idx, 0] = gpu_t
                gpu_v_photodiode[synch_idx, 1] = diode_time[argmin]
        
        reg = LinearRegression().fit(gpu_v_photodiode[:,0].reshape(-1, 1), gpu_v_photodiode[:,1])
        r2 = reg.score(gpu_v_photodiode[:,0].reshape(-1, 1), gpu_v_photodiode[:,1])
        slope = reg.coef_[0]
        intercept = reg.intercept_
        print(f"R2={r2} ---- Slope={slope} ---- intercept={intercept}")
        gpu_2_photodiode_time = lambda gpu_time: gpu_time * slope + intercept # Define functions for time conversion

        # # #  combine events if thy happen on the same frame
        events_csv_data['EventKey'] = events_csv_data['Frame.Time'].astype(str) + '_' + events_csv_data['EventName']
        events_aggregated = events_csv_data.groupby(['Frame.Index', 'Frame.Time']).agg({'EventName': lambda x: ', '.join(x),'EventData': lambda x: ', '.join(x.astype(str))}).reset_index()
        averaged_vr_data = vr_csv_data.groupby('Index', as_index=False)['Position'].mean()
        averaged_encoder_data = encoder_csv_data.groupby('FrameIndex', as_index=False)['RotaryEncoder'].mean()
        merged_data = counter_csv_data.merge( events_aggregated[['Frame.Index', 'EventName', 'EventData']], left_on='Index',  right_on='Frame.Index',  how='left')
        merged_data = merged_data.merge(averaged_vr_data, left_on='Index', right_on='Index',how='left')
        merged_data = merged_data.merge(averaged_encoder_data, left_on='Index', right_on='FrameIndex',how='left')
        merged_data.rename(columns={'Position': 'Averaged_Position', 'RotaryEncoder': 'Averaged_Encoder'}, inplace=True)
        
        m_per_tick = 0.012
        dt = merged_data["Time"].diff().to_numpy()
        dc = merged_data["Averaged_Encoder"].diff().to_numpy()
        v_m_s = (dc * m_per_tick) / dt
        speed_m_s = np.abs(v_m_s)
        speed_m_s = np.nan_to_num(speed_m_s, nan=0.0, posinf=0.0, neginf=0.0)

        stop_thresh_m_s = 0.01
        speed_m_s[speed_m_s < stop_thresh_m_s] = 0.0

        merged_data["Speed_Absolute"] = speed_m_s
        merged_data['Reward'] = merged_data['EventName'].str.contains('Reward', case=False, na=False).astype(int)
        merged_data['Lick'] = merged_data['EventName'].str.contains('Lick', case=False, na=False).astype(int)
        merged_data['Teleport'] = merged_data['EventName'].str.contains('Teleport', case=False, na=False).astype(int)
        merged_data['Stim'] = merged_data['EventName'].str.contains('WallVisibility', case=False, na=False).astype(int)
        merged_data['Opto']= merged_data['EventName'].str.contains('OptoStim', case=False, na=False).astype(int)
        event_data_list = merged_data['EventData'][merged_data[merged_data['Stim']==1].index].tolist()
        stim_value = []
        for event in event_data_list:
            value = int(event.split('L')[1].split('R')[0])
            stim_value.append(value)
        merged_data.loc[merged_data[merged_data['Stim']==1].index, 'Stim'] = stim_value
        merged_data = merged_data.drop(columns=['EventName', 'EventData', 'Frame.Index'])
        stim_times = merged_data.loc[merged_data['Stim'] > 0, 'Time'].values
        merged_data['Time'] = gpu_2_photodiode_time(merged_data['Time'].values)  
        
        piezo_signal = analog_data[1 if analog_data.shape[0] > 2 else 0, :]
        piezo_time = np.arange(piezo_signal.shape[0]) / fs_analog
        indices = np.searchsorted(piezo_time, merged_data['Time'])
        indices = np.clip(indices, 0, len(piezo_signal) - 1)
        merged_data['Piezo_Signal'] = piezo_signal[indices]
        trial_start_indices_full = merged_data[merged_data['Teleport'] == 1].index.tolist()
        reward_indices_full = merged_data[merged_data['Reward'] == 1].index.tolist()
        pupil_data = pupil_data.copy()
        pupil_data["Frames"] = np.arange(len(pupil_data))
        pupil_data["Time"] = pupil_data["Frames"] / 30.0
        data_path = os.path.join(self.behavior_data_path, 'data.pkl')
        expected_n_volumes = len(scanner_time[::n_planes])
        if not os.path.exists(data_path):
            print("data not found. Processing data...")
            downsampled=self.decimate_dataframe(merged_data,scanner_time,n_planes) 
        else:
            downsampled = pd.read_pickle(data_path)
            if len(downsampled) != expected_n_volumes:
                downsampled = self.decimate_dataframe(merged_data, scanner_time, n_planes)

        pupil_cols = {"pupil_area", "pupil_diameter"} 
        has_pupil = pupil_cols.intersection(downsampled.columns)
        has_pupil = None
        if has_pupil:
            print("Pupil metrics already present -> skipping DLC processing.")
        else:
            dlc_path = self.find_pupil_dlc_output(self.behavior_data_path)
            if dlc_path is None:
                print("No DLC pupil output found -> proceeding without pupil metrics.")
            else:
                print("Using DLC file:", Path(dlc_path).name)
                try:
                    x, y, p = self.load_dlc_points(dlc_path)
                    T = x.shape[0]
                    bin_starts = scanner_time[::n_planes]
                    t0, t1 = bin_starts[0], bin_starts[-1]
                    cam = np.asarray(camera_time)
                    j0 = np.searchsorted(cam, t0)

                    if T > j0:
                        x, y, p = x[j0:], y[j0:], p[j0:]
                    else:
                        raise ValueError("DLC has fewer frames than camera pulses before scanner start.")

                    T_eff = min(x.shape[0], len(cam) - j0)
                    x, y, p = x[:T_eff], y[:T_eff], p[:T_eff]
                    cam_frame_time = cam[j0:j0 + T_eff]

                    x_d, y_d, p_d = self.bin_and_average_dlc(
                        x, y, p, cam_frame_time, bin_starts
                    )

                    pupil_metrics = self.extract_pupil_from_dlc(
                        x_d, y_d, p_d,
                        jump_thresh=20,
                        min_points=5, min_coverage = 0.5,
                    )

                    for k, v in pupil_metrics.items():
                        downsampled[k] = v

                    bin_idx = np.searchsorted(bin_starts, cam_frame_time, side="right") - 1
                    counts = np.bincount(
                        bin_idx[(bin_idx >= 0) & (bin_idx < len(bin_starts))],
                        minlength=len(bin_starts)
                    )

                except Exception as e:
                    print("WARNING: Failed to compute pupil metrics:", repr(e))

        downsampled.to_pickle(os.path.join(self.behavior_data_path, "data.pkl"))

        return downsampled, pupil_data, merged_data['Piezo_Signal'], merged_data['Time'], trial_start_indices_full, reward_indices_full, scanner_time_is_reconstructed, scanner_time_diagnostics
    
    def bin_and_average_dlc(self, x, y, p, cam_frame_time, bin_starts):
        T, n_pts = x.shape
        N = len(bin_starts)

        bin_idx = np.searchsorted(bin_starts, cam_frame_time, side="right") - 1
        counts = np.bincount(bin_idx[(bin_idx>=0)&(bin_idx<len(bin_starts))], minlength=len(bin_starts))

        valid = (bin_idx >= 0) & (bin_idx < N)
        if not np.any(valid):
            return (np.full((N, n_pts), np.nan),
                    np.full((N, n_pts), np.nan),
                    np.full((N, n_pts), np.nan))

        bin_idx = bin_idx[valid]
        x = x[valid]
        y = y[valid]
        p = p[valid]

        x_d = np.full((N, n_pts), np.nan)
        y_d = np.full((N, n_pts), np.nan)
        p_d = np.full((N, n_pts), np.nan)

        for i in np.unique(bin_idx):
            m = bin_idx == i
            x_d[i] = np.nanmean(x[m], axis=0)
            y_d[i] = np.nanmean(y[m], axis=0)
            p_d[i] = np.mean(p[m] >= 0.97, axis=0)
        return x_d, y_d, p_d

    def decimate_dataframe(self, df, scanner_times, nz, time_col_name='Time', binary_cols=['Lick','Reward','Teleport','Opto'], event_cols=['Stim','Teleport','Opto']):
        start_idx = np.searchsorted(df[time_col_name].values, scanner_times)
        start_idx = start_idx[::nz]
        stop_idx = np.append(start_idx[1:] - 1, len(df) - 1)
        for i in range(1, len(start_idx)): 
            if start_idx[i] == start_idx[i-1]:
                stop_idx[i] = stop_idx[i-1] + 1
            elif stop_idx[i] <= start_idx[i]: 
                stop_idx[i] = start_idx[i] + 1  
        stop_idx[-1] = len(df) - 1

        resampled_data = []
        for start, stop in zip(start_idx, stop_idx):
            chunk = df.iloc[start:stop + 1]
            avg_data = {}
            for col in df.columns:
                if col == time_col_name:
                    if chunk[col].isna().sum() > 0:
                        avg_data[col] = np.nan  
                    else:
                        avg_data[col] = chunk[col].mean()  
                elif col in binary_cols:
                    avg_data[col] = int(chunk[col].sum() > 0)
                elif col in event_cols:
                    avg_data[col] = chunk[col].max()
                else:
                    avg_data[col] = np.nanmean(chunk[col])
            resampled_data.append(avg_data)

        decimated_df = pd.DataFrame(resampled_data)
        if decimated_df['Time'].isna().sum().sum() > 0:
            print(f"Warning: NaNs found in the resampled dataframe")
        return decimated_df
    
    def align_frames(self,downsampled_data, dff_Zscore):
        if len(downsampled_data) != dff_Zscore.shape[1]:
            excess_frames = len(downsampled_data) - dff_Zscore.shape[1]
            if excess_frames > 0:
                print(f"Detected {excess_frames} excess frames in the downsampled data. Discarding these frames.")
                downsampled_data = downsampled_data.iloc[:-excess_frames]
            else:
                missing_frames = -excess_frames
                print(f"WARNING: behavior logging stopped before the "
                      f"scanner did). Discarding the {missing_frames} trailing "
                      f"imaging frames that have no corresponding behavior data. ")
                dff_Zscore = dff_Zscore[:, :len(downsampled_data)]
        return downsampled_data.reset_index(drop=True), dff_Zscore
    
    def align_data(self, downsampled_data, dff_Zscore):
        aligned_data, dff_Zscore = self.align_frames(downsampled_data, dff_Zscore)
        gratings_start = aligned_data[~aligned_data['Stim'].isin([0, np.nan])].index
        gratings_start = gratings_start[np.r_[True, np.diff(gratings_start) > 2]]
        trial_start_indices = aligned_data[aligned_data['Teleport'] == 1].index.tolist()
        opto_stim = aligned_data[aligned_data['Opto'] == 1].index.tolist()
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
        log_file = os.path.join(self.behavior_data_path,'TrialLogging.yaml')
        trial_types = np.array(self.extract_trial_type(log_file,'trialTypeLabel')[1:])
        
        n_trials = max(len(trial_start_indices) - 1, 1)
        n_gratings = round(len(gratings_start) / n_trials)
        grating_onsets_dict = {f'gr_{i+1}': [] for i in range(n_gratings)}
        for i, idx in enumerate(gratings_start):
            grating_num = (i % n_gratings) + 1
            grating_onsets_dict[f'gr_{grating_num}'].append(idx)
        
        pre_frames = 20
        post_frames = 40
        total_frames = pre_frames + post_frames

        time_values = aligned_data['Time'].values
        dt = 7.9 / total_frames  
        grating_indices = {}

        for grating, onset_indices  in grating_onsets_dict.items():
            expanded = []
            for onset_idx in onset_indices:
                onset_time = aligned_data['Time'].iloc[onset_idx]
                target_times = onset_time + (np.arange(total_frames) - pre_frames) * dt
                idxs = np.searchsorted(time_values, target_times)
                idxs = np.clip(idxs, 0, len(time_values) - 1)
                expanded.append(idxs)

            grating_indices[grating] = expanded
            
        post_frames = pre_frames
        total_frames = pre_frames + post_frames

        time_values = aligned_data["Time"].values
        dt = np.nanmedian(np.diff(time_values))

        reward_indices = []
        for ridx in reward_delivery:
            reward_time = aligned_data["Time"].iloc[ridx]
            target_times = reward_time + (np.arange(total_frames) - pre_frames) * dt
            idxs = np.searchsorted(time_values, target_times)
            idxs = np.clip(idxs, 0, len(time_values) - 1)
            reward_indices.append(idxs)
        
        n_trial_types = trial_types.max()+1
        tt_mat = np.zeros((n_trials, n_trial_types), dtype=int)
        tt_mat[np.arange(n_trials), trial_types[:n_trials]] = 1

        unpred_trials = {f'gr_{i+1}': [] for i in range(n_gratings)}
        all_stimuli = set(np.unique(aligned_data['Stim'].dropna()))
        expected_stim_set = {2, 4}
        unpred_stim_set = all_stimuli - expected_stim_set
        pred_trials = []

        for gr, indices in grating_onsets_dict.items():
            for i, idx in enumerate(indices):
                if aligned_data.loc[idx, 'Stim'] in unpred_stim_set:
                    unpred_trials[gr].append(i)
                else:
                    pred_trials.append(i)
        return n_gratings,aligned_data, grating_indices, reward_indices, n_trials, tt_mat, trial_start_indices, pred_trials, unpred_trials, dff_Zscore
    
class Suite2pLoader:
    def process_suite2p_data(self, suite2ppath, neuropil_factor,roitype):
        ops = np.load(os.path.join(suite2ppath, f'plane0','ops.npy'), allow_pickle=True).item()
        dF_F = []
        dF_F_red = []
        all_cells_indices = []
        n_planes = ops['nplanes']
        
        all_TC = []
        for plane_num in range(n_planes):
            plane_path = os.path.join(suite2ppath, f'plane{plane_num}')
            iscell = np.load(os.path.join(plane_path,'iscell.npy'))
            cells_indices = np.where(iscell[:, 0])[0]

            if roitype == 'manual_suite2p':
                F = np.load(os.path.join(plane_path, 'F_red.npy'))
                Fneu = np.load(os.path.join(plane_path, 'Fneu_red.npy'))
                red_path = plane_path.replace(r'\suite2p', r'\\red\suite2p')
                iscell = np.load(os.path.join(red_path, 'iscell.npy'))
                cells_indices = np.where(iscell[:, 0])[0]
                F = F[cells_indices,:]
                Fneu = Fneu[cells_indices, :]
                TC_plane = (F) - (neuropil_factor * (Fneu))
            
            elif roitype == 'manual_imagej':
                F = np.load(os.path.join(plane_path, 'F_red.npy')) 
                Fneu = np.load(os.path.join(plane_path, 'Fneu_red.npy')) 
                TC_plane = (F) - (neuropil_factor * (Fneu))
            
            elif roitype == 'suite2p':
                F = np.load(os.path.join(plane_path, 'F.npy')) 
                Fneu = np.load(os.path.join(plane_path, 'Fneu.npy')) 
                F = F[cells_indices,:]
                Fneu = Fneu[cells_indices, :]
                TC_plane = (F) - (neuropil_factor * (Fneu))
            elif roitype == 'aligned':
                try:
                    F = np.load(os.path.join(plane_path, 'F_aligned.npy'))
                    TC_plane = F
                except FileNotFoundError:
                    continue

            all_TC.append(TC_plane.T) 
            all_cells_indices.append(cells_indices)
            try:
                F_red = np.load(os.path.join(plane_path, 'F_chan2.npy'))
                TC_red = F_red[cells_indices,:]
                TC_red_smoothed = gaussian_filter1d(TC_red, sigma=1, axis=1)
                baseline_red = np.percentile(TC_red_smoothed, 50, axis=1)
                dF_red = TC_red_smoothed - baseline_red[:, None]
                TC_red_dff = dF_red.T / baseline_red
                absolute_fluorescence_red = TC_red_smoothed
                dF_F_red.append(absolute_fluorescence_red)
            except FileNotFoundError:
                pass
        
        max_timepoints = max(tc.shape[0] for tc in all_TC)
        
        for i in range(len(all_TC)):
            if all_TC[i].shape[0] < max_timepoints:
                padding_size = max_timepoints - all_TC[i].shape[0]
                all_TC[i] = np.vstack([all_TC[i], np.zeros((padding_size, all_TC[i].shape[1]))])
        
        TC_combined = np.hstack(all_TC)  
        TC_smoothed = gaussian_filter1d(TC_combined, sigma=0.7, axis=0) 
        baseline = np.percentile(TC_smoothed, 50, axis=0)  
        TC_dff = (TC_smoothed - baseline) / baseline
        
        dF_F_Zscore = (TC_dff - np.nanmean(TC_dff, axis=0)) / np.nanstd(TC_dff, axis=0)
        dF_F = dF_F_Zscore.T
        
        if dF_F_red:
            max_columns = dF_F.shape[1]  
            for i in range(len(dF_F_red)):
                if dF_F_red[i].shape[1] < max_columns:
                    padding_size = max_columns - dF_F_red[i].shape[1]
                    dF_F_red[i] = np.hstack([dF_F_red[i], np.full((dF_F_red[i].shape[0], padding_size), np.nan)])
            dF_F_red = np.vstack(dF_F_red)
        flat_map = []
        for plane_num, cell_ids in enumerate(all_cells_indices):
            for roi_id in cell_ids:
                flat_map.append((plane_num, roi_id))

        return ops, dF_F, all_cells_indices, n_planes, iscell, TC_combined.T, dF_F_red

class DenoiseData():
    def __init__(self, grating_indices):
        self.grating_indices = grating_indices
    
    def baseline_subtraction(self, zscore, zscore_red, trial_start_indices):
        trial_start_frames = 15
        for trial_idx in range(len(trial_start_indices) - 1): 
            start_idx = trial_start_indices[trial_idx] 
            end_idx = trial_start_indices[trial_idx + 1] 
            baseline = np.mean(zscore[start_idx : start_idx + trial_start_frames], axis=0)
            zscore[start_idx:end_idx] -= baseline

        if zscore_red is not None and len(zscore_red) > 0:
            for trial_idx in range(len(trial_start_indices) - 1):
                start_idx = trial_start_indices[trial_idx]
                end_idx = trial_start_indices[trial_idx + 1]
                baseline_red = np.median(zscore_red[start_idx : start_idx + trial_start_frames], axis=0)
                zscore_red[start_idx:end_idx] -= baseline_red

        return zscore, zscore_red

    def filter_trials(self, filter_val, act_trials, act_red_trials, trials):
            stability = {}
            true_indices = {}
            good_trials = {}
            activity_trials={}
            activity_red_trials={}
            trials=trials[:-1]
            rois = act_trials['gr_1'].shape[0]
            for _, (grat, _) in enumerate(self.grating_indices.items()):
                stability[grat] = np.ones((rois, len(trials)))
                activity_trials[grat]=act_trials[grat]
                activity_trials[grat] = activity_trials[grat][:,trials,:]
                activity_red_trials[grat]=act_red_trials[grat]
                activity_red_trials[grat] = activity_red_trials[grat][:,trials,:]
                for roi, _ in enumerate(activity_trials[grat]):
                    tdtomsig = np.nanmean(activity_red_trials[grat][roi, :, slice(23,34)], axis=1)
                    gcampsig = np.nanmean(activity_trials[grat][roi,:, slice(23,34)], axis=1)
                    X = np.arange(len(tdtomsig))
                    slope, intercept = np.polyfit(X, tdtomsig, 1)
                    predicted_values = intercept + slope * X
                    noise_dist = np.abs(tdtomsig - predicted_values)
                    mean_val = np.mean(noise_dist)
                    std_dev = np.std(noise_dist)
                    z_scores = np.abs((noise_dist - mean_val) / std_dev)
                    
                    outlier = 1.0  
                    stability[grat][roi, np.where(z_scores > outlier)[0]] = 0
                    
                true_indices[grat] = np.where(np.sum(stability[grat], axis=0) / rois > filter_val)[0]
                good_trials[grat] = np.array(trials)[true_indices[grat]]
            
            return stability, good_trials, true_indices

class  DataProcessor:
    def __init__(self):
        self.n_gratings = None
        self.trial_start_indices = None
        self.grating_indices = None
        self.position = None
        self.position_tunnel = None
        self.trial_types = None
        self.n_trials = None
        self.dff_Zscore = None
        self.dff_Zscore_red = None
        self.n_planes = None
        self.aligned_data = None
        self.reward_indices = None
        self.log_file = None
        self.denoiser = None
        self.unpred_trials = None
        self.pred_trials = None
        self.unpred_led = None
        self.unpred_noled = None
        self.catchB = None
        self.catchA = None
        self.activity = {}
        self.speed = None
        self.lick = None
        self.pupil = None
        self.piezo_full = None
        self.time_full = None
        self.trial_start_indices_full = None
        self.reward_indices_full = None
        self.scanner_time_is_reconstructed = False
        self.scanner_time_diagnostics = None

    def preprocessing(self, table, sheet_name, neuropil_factor, ani, tri_perc, suite2ppath, behavior_data_path,basesub, roitype):
        data_loader = DataLoader(behavior_data_path, sheet_name, suite2ppath)
        suite2p_loader = Suite2pLoader()
        if roitype == 'manual':
            roitype = table.iloc[ani][10]

        if table.iloc[ani][1] == 1:
            matlab_data = data_loader.load_matlab_data()
            self.n_gratings, self.unpred_trials, self.pred_trials, self.trial_start_indices, self.grating_indices, self.reward_indices, self.position, num_bins, self.position_tunnel, \
                self.n_trials, self.unpred_led, self.unpred_noled, self.catchA, self.catchB, self.speed, self.lick, self.pupil = data_loader.load_matlab_variables(matlab_data)
            ops, self.dff_Zscore, all_cells_indices, self.n_planes, iscell,self.TC, self.dff_Zscore_red = suite2p_loader.process_suite2p_data(suite2ppath,neuropil_factor,roitype)
            print('matlab done')

        else:  
            analog_time, analog_data, fs_analog = data_loader.load_analog_data()
            digital_time, digital_data, fs_digital = data_loader.load_digital_data()

            ops, self.dff_Zscore, all_cells_indices, self.n_planes, iscell,self.TC, self.dff_Zscore_red = suite2p_loader.process_suite2p_data(suite2ppath, neuropil_factor, roitype)
            encoder_csv_data, vr_csv_data, events_csv_data, counter_csv_data, quadsync_csv_data, self.log_file, pupil_data = data_loader.load_csv_data()
            n_frames_expected = self.dff_Zscore.shape[1] # frame count / rate from Suite2p metadata if the scanner DAQ is missing for this session.
            fs_expected = ops.get('fs', None)
            downsampled_data, all_pupil_data,self.piezo_full, self.time_full, self.trial_start_indices_full, self.reward_indices_full, self.scanner_time_is_reconstructed, self.scanner_time_diagnostics = data_loader.align_events(digital_data, counter_csv_data, events_csv_data, quadsync_csv_data, vr_csv_data, encoder_csv_data,pupil_data, analog_data, fs_analog, fs_digital, self.n_planes, n_frames_expected=n_frames_expected, fs_expected=fs_expected)
            self.n_gratings,self.aligned_data, self.grating_indices, self.reward_indices, self.n_trials, self.trial_types, self.trial_start_indices, self.pred_trials, self.unpred_trials, self.dff_Zscore = data_loader.align_data(downsampled_data, self.dff_Zscore)
            n_frames_aligned = self.dff_Zscore.shape[1]
            if self.dff_Zscore_red is not None and len(self.dff_Zscore_red) > 0 and self.dff_Zscore_red.shape[1] != n_frames_aligned:
                self.dff_Zscore_red = self.dff_Zscore_red[:, :n_frames_aligned]
            if self.TC is not None and self.TC.shape[0] != n_frames_aligned:
                self.TC = self.TC[:n_frames_aligned, :]
            print('bonsai done')

        self.denoiser = DenoiseData(self.grating_indices)
        if basesub == 1:
            self.dff_Zscore, self.dff_Zscore_red = self.denoiser.baseline_subtraction(self.dff_Zscore, self.dff_Zscore_red, self.trial_start_indices)

        for gr, _ in self.grating_indices.items():
            self.activity[gr] = self.dff_Zscore[:,self.grating_indices[gr]]


def main(ani, table,sheet_name, neuropil_factor, tri_perc, basesub = 0, roitype = 'suite2p'):
    suite2ppath = table.iloc[ani][5]  
    behavior_data_path = table.iloc[ani][2]
    data_processor = DataProcessor()
    data_processor.preprocessing(table, sheet_name, neuropil_factor, ani, tri_perc, suite2ppath, behavior_data_path, basesub, roitype)

    if data_processor.scanner_time_is_reconstructed:
        print(f"NOTE: animal index {ani} used pupil-based scanner_time reconstruction "
              f"(scanner DAQ channel was missing). Diagnostics: "
              f"{data_processor.scanner_time_diagnostics}")

    return ani, data_processor

if __name__ == "__main__":
    main()