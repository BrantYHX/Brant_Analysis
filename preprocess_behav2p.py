# MODULES
import numpy as np
import pandas as pd
import os
import yaml
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.io

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

        unpred_trials = struct_to_dict(mat_data['unpred_trials'])
        for gr in range(1, n_gratings + 1):
            key = f'gr_{gr}'
            unpred_trials[key] = unpred_trials.get(key, []).tolist() if isinstance(unpred_trials.get(key, []), np.ndarray) else unpred_trials.get(key, [])

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

        speed = mat_data['speed'].squeeze()
        if np.isnan(mat_data['pupil']):
            pupil = []
        else:
            pupil = mat_data['pupil'].squeeze()
        
        return n_gratings,unpred_trials, pred_trials, trial_start_indices, grating_indices, position, num_bins, position_tunnel, n_trials, unpred_led, unpred_noled, catchA, catchB, speed

    @staticmethod    
    def extract_digital_channel(digital_data, channel_number):
        #channel_number goes from 0 to 7
        return digital_data & (1 << channel_number) > 0
    
    @staticmethod
    def detect_edges(signal):
        edges =np.where(np.diff(signal) == 1)[0] +1# identify where in time frame signal
        return edges
    
    @staticmethod
    def detect_falling_edges(signal):
        edges = np.where((np.array(signal[:-1]) == True) & (np.array(signal[1:]) == False))[0] + 1
        return edges

    @staticmethod
    def find_nearest_index(time_array, target):
        idx = np.searchsorted(time_array, target)
        if idx == 0:
            return 0##
        if idx >= len(time_array):
            return len(time_array) - 1
        before = time_array[idx - 1]
        after = time_array[idx]
        return idx - 1 if abs(before - target) < abs(after - target) else idx
                
    def load_excel_data(self):
        return pd.read_excel(self.behavior_data_path, sheet_name=self.sheet_name)
    
    def load_matlab_data(self):
        return scipy.io.loadmat(os.path.join(self.suite2ppath, 'data.mat'), struct_as_record=False, squeeze_me=True)
    
    def load_analog_data(self, fs_analog=1000):
        analog_file = os.path.join(self.behavior_data_path, 'AnalogInput.bin')
        analog_data = np.fromfile(analog_file, dtype=np.float64).reshape(-1, 2).transpose()
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
    
    def align_events(self, digital_data, counter_csv_data, events_csv_data, quadsync_csv_data, vr_csv_data, encoder_csv_data, pupil_data,analog_data, fs_analog, fs_digital, n_planes):
        # Align synchronization events between the GPU and the photodiode signal
        photodiode_signal = self.extract_digital_channel(digital_data, 2)
        gpu_toggle_time = quadsync_csv_data["Time"].values
        diode_time = self.detect_edges(photodiode_signal) / fs_digital
        min_nidaq_sample = 10_000
        try:
            first_diode_time = diode_time[diode_time > (min_nidaq_sample / fs_digital)][0]
        except IndexError:
            print("WARNING: No photodiode data detected. Using predicted timestamps.")
            slope_avg = 1.0000184161
            intercept_avg = 1.511616
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

        # Load Pupil metadata
        pupil_data['Frames'] = range(0, len(pupil_data['Item1']) )
        pupil_data['Time'] = pupil_data['Frames'] /30  # sampling freq of camera
        if os.path.exists(os.path.join(self.behavior_data_path, 'PupilCamera_CameraVideoDLC_resnet50_pupil_analysis_newNov20shuffle1_170000_filtered.csv')):
            dlc_data = pd.read_csv(os.path.join(self.behavior_data_path, 'PupilCamera_CameraVideoDLC_resnet50_pupil_analysis_newNov20shuffle1_170000_filtered.csv'))
            dlc_data = dlc_data.iloc[2:, 1:]
            pupil_data.reset_index(drop=True, inplace=True)
            dlc_data.reset_index(drop=True, inplace=True)
            all_pupil_data = pd.concat([pupil_data, dlc_data], axis=1)
        else:
            all_pupil_data = pupil_data

        # extract scanner times
        scanner_signal = self.extract_digital_channel(digital_data, 0)
        scanner_time = self.detect_falling_edges(scanner_signal) / fs_digital # the end of a scanning cycle or acquisition frame.
        #  combine events if thy happen on the same frame
        events_csv_data['EventKey'] = events_csv_data['Frame.Time'].astype(str) + '_' + events_csv_data['EventName']
        events_aggregated = events_csv_data.groupby(['Frame.Index', 'Frame.Time']).agg({'EventName': lambda x: ', '.join(x),'EventData': lambda x: ', '.join(x.astype(str))}).reset_index()
        # events_aggregated = events_csv_data.groupby(['Frame.Index', 'Frame.Time']).agg({'EventName': lambda x: ', '.join(x), 'EventData': lambda x: ', '.join(x)}).reset_index() # combine events
        averaged_vr_data = vr_csv_data.groupby('Index', as_index=False)['Position'].mean()#  average vr positions if they occur on the same frames:
        averaged_encoder_data = encoder_csv_data.groupby('FrameIndex', as_index=False)['RotaryEncoder'].mean()#  average vr positions if they occur on the same frames:
        #  merge with frame counters
        merged_data = counter_csv_data.merge( events_aggregated[['Frame.Index', 'EventName', 'EventData']], left_on='Index',  right_on='Frame.Index',  how='left')
        merged_data = merged_data.merge(averaged_vr_data, left_on='Index', right_on='Index',how='left')
        merged_data = merged_data.merge(averaged_encoder_data, left_on='Index', right_on='FrameIndex',how='left')
        merged_data.rename(columns={'Position': 'Averaged_Position', 'RotaryEncoder': 'Averaged_Encoder'}, inplace=True)
        
        wheel_dim = 0.157
        merged_data['Speed_Absolute'] = (merged_data['Averaged_Encoder'].diff() * wheel_dim) / merged_data[
            'Time'].diff()
        merged_data['Speed_Absolute'].fillna(0, inplace=True)

        merged_data['Reward'] = merged_data['EventName'].str.contains('Reward', case=False, na=False).astype(int)
        merged_data['Lick'] = merged_data['EventName'].str.contains('Lick', case=False, na=False).astype(int)
        merged_data['Teleport'] = merged_data['EventName'].str.contains('Teleport', case=False, na=False).astype(int)
        merged_data['Stim'] = merged_data['EventName'].str.contains('WallVisibility', case=False, na=False).astype(int)
        event_data_list = merged_data['EventData'][merged_data[merged_data['Stim']==1].index].tolist()
        stim_value = []
        for event in event_data_list:
            value = int(event.split('L')[1].split('R')[0])
            stim_value.append(value)
        merged_data.loc[merged_data[merged_data['Stim']==1].index, 'Stim'] = stim_value
        merged_data = merged_data.drop(columns=['EventName', 'EventData', 'Frame.Index'])

        piezo_signal = analog_data[1, :]
        piezo_time = np.arange(piezo_signal.shape[0]) / fs_analog
        indices = np.searchsorted(piezo_time, merged_data['Time'])
        indices = np.clip(indices, 0, len(piezo_signal) - 1)  #ensure within valid range for indices
        merged_data['Piezo_Signal'] = piezo_signal[indices]
        merged_data['Time'] = gpu_2_photodiode_time(merged_data['Time'].values)    
        # plt.figure()
        # plt.plot(diode_time)
        # # plt.show()
        
        # plt.title('photodiode_time')
        # # sync events should be aligned for gpu and photodiode
        # plt.figure()
        # plt.plot(np.diff(gpu_v_photodiode[:,1]), label="Photodiode", lw=3)
        # plt.plot(np.diff(gpu_v_photodiode[:,0]), label="GPU")

        # plt.title("$\Delta$Time between adjacent sync events")
        # plt.xlabel("Synch Event")
        # plt.ylabel("Time(s)")
        # plt.legend()
        # plt.show()

        # plt.figure()
        # plt.hist(gpu_v_photodiode[:,1]-gpu_v_photodiode[:,0])
        # # Calculate the regression that takes you Time(GPU) -> Time(NIDAQ)
        # plt.figure()
        # plt.scatter(gpu_v_photodiode[:,0], gpu_v_photodiode[:,1])
        # plt.title("GPU vs Photodiode time")
        # plt.xlabel("Time(s)")
        # plt.ylabel("Time(s)")
        # # plt.show()

        # predicted_photodiode_times = reg.predict(gpu_v_photodiode[:, 0].reshape(-1, 1))
        # residuals = gpu_v_photodiode[:, 1] - predicted_photodiode_times
        # plt.figure()
        # plt.scatter(gpu_v_photodiode[:, 0], residuals)
        # plt.axhline(0, color='r', linestyle='--')
        # plt.title("Residuals of GPU to Photodiode Time Regression")
        # plt.xlabel("GPU Time(s)")
        # plt.xlim([0,3000])
        # plt.ylabel("Residuals (s)")
        # plt.show()

        # plt.figure(figsize=(10, 6))
        # time_differences = gpu_v_photodiode[:,1] - gpu_v_photodiode[:,0]
        # mean_diff = np.mean(time_differences)
        # std_diff = np.std(time_differences)

        # # Use more bins for better resolution and add styling
        # n, bins, patches = plt.hist(time_differences, bins=30, color='skyblue', edgecolor='black', alpha=0.7)

        # # Add vertical line at mean
        # plt.axvline(mean_diff, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_diff:.6f}s')

        # # Add title and labels
        # plt.title('Distribution of Time Differences (Photodiode - GPU)', fontsize=14)
        # plt.xlabel('Time Difference (seconds)', fontsize=12)
        # plt.ylabel('Frequency', fontsize=12)

        # # Add statistics annotation
        # plt.annotate(f'Mean: {mean_diff:.6f}s\nStd Dev: {std_diff:.6f}s', 
        #             xy=(0.7, 0.85), xycoords='axes fraction',
        #             bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

        # plt.grid(alpha=0.3)
        # plt.tight_layout()
        # plt.show()



        # plt.figure(figsize=(12, 5))
        # plt.plot(gpu_v_photodiode[:,0], time_differences, 'o-', markersize=3)
        # plt.axhline(mean_diff, color='red', linestyle='--', label=f'Mean difference: {mean_diff:.6f}s')
        # plt.fill_between(gpu_v_photodiode[:,0], mean_diff-std_diff, mean_diff+std_diff, color='red', alpha=0.2, label=f'±1 std dev: {std_diff:.6f}s')
        # plt.title('Time Difference (Photodiode - GPU) Over Recording Duration', fontsize=14)
        # plt.xlabel('GPU Time (s)', fontsize=12)
        # plt.ylabel('Time Difference (s)', fontsize=12)
        # plt.legend()
        # plt.grid(alpha=0.3)
        # plt.tight_layout()
        # plt.show()
        # plt.figure(figsize=(12, 5))
        # plt.subplot(1, 2, 1)
        # plt.hist(time_differences, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        # plt.title('Raw Time Differences\n(Photodiode - GPU)', fontsize=12)
        # plt.xlabel('Time Difference (s)', fontsize=10)
        # plt.ylabel('Frequency', fontsize=10)
        # plt.grid(alpha=0.3)

        # # Residuals after regression
        # plt.subplot(1, 2, 2)
        # plt.hist(residuals, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        # plt.title(f'Residuals After Linear Regression\nR² = {r2:.4f}', fontsize=12)
        # plt.xlabel('Residual (s)', fontsize=10)
        # plt.ylabel('Frequency', fontsize=10)
        # plt.grid(alpha=0.3)

        # plt.tight_layout()
        # plt.show()
        downsampled_data = self.check_data(merged_data,scanner_time,n_planes)

        return downsampled_data, all_pupil_data
    
    def check_data(self, data,scanner_times,n_planes):
        data_path = os.path.join(self.behavior_data_path, 'data.pkl')
        pupil_csv = os.path.join(self.behavior_data_path, 'PupilCamera_CameraVideoDLC_resnet50_pupil_analysis_newNov20shuffle1_170000_filtered.csv')

        if not os.path.exists(data_path):
            print("data not found. Processing data...")
            downsampled_data=self.decimate_dataframe(data,scanner_times,n_planes) 
            downsampled_data.to_pickle(os.path.join(self.behavior_data_path, 'data.pkl')) #save data
        elif os.path.exists(data_path) and not os.path.exists(pupil_csv):
            print("data found, no dlc pupil_data yet. Loading existing data...")
            # downsampled_data = pd.read_pickle(data_path)
            downsampled_data=self.decimate_dataframe(data,scanner_times,n_planes) 
            downsampled_data.to_pickle(os.path.join(self.behavior_data_path, 'data.pkl')) #save data

        elif os.path.exists(data_path) and os.path.exists(pupil_csv) and not os.path.exists(os.path.join(self.behavior_data_path, 'pupil_data.pkl')):
            print("new pupil data found. Reprocess data...")
            downsampled_data=self.decimate_dataframe(data,scanner_times,n_planes) 
            downsampled_data.to_pickle(os.path.join(self.behavior_data_path, 'data.pkl')) #save data
        elif os.path.exists(data_path) and os.path.exists(os.path.join(self.behavior_data_path, 'pupil_data.pkl')):
            print("all data found. Loading existing data...")
            downsampled_data=self.decimate_dataframe(data,scanner_times,n_planes) 
            downsampled_data.to_pickle(os.path.join(self.behavior_data_path, 'data.pkl')) #save data
            downsampled_data = pd.read_pickle(data_path)
        return downsampled_data
    
    def decimate_dataframe(self, df, scanner_times, nz, time_col_name='Time', binary_cols=['Lick','Reward','Teleport'], event_cols=['Stim']):
        start_idx = np.searchsorted(df[time_col_name].values, scanner_times) # align df timestamp to nearest falling edge of scanner frame
        start_idx = start_idx[::nz]  # take every nz-th start index
        stop_idx = np.append(start_idx[1:] - 1, len(df) - 1)
        for i in range(1, len(start_idx)): 
            if start_idx[i] == start_idx[i-1]:
                stop_idx[i] = stop_idx[i-1] + 1
            elif stop_idx[i] <= start_idx[i]: 
                stop_idx[i] = start_idx[i] + 1  
        stop_idx[-1] = len(df) - 1 # change the last stop_idx to the end of the data

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
        # Check if the number of detected frames matches the number of frames in roi_array
        if len(downsampled_data) != dff_Zscore.shape[1]:
            excess_frames = len(downsampled_data) - dff_Zscore.shape[1]
            if excess_frames > 0:
                print(f"Detected {excess_frames} excess frames in the downsampled data. Discarding these frames.")
                downsampled_data = downsampled_data.iloc[:-excess_frames]
            else:
                raise ValueError("Mismatch between detected frames in scanner signal and ROI data frames.")
        return downsampled_data
    
    def align_data(self, downsampled_data, dff_Zscore):

        aligned_data = self.align_frames(downsampled_data, dff_Zscore)
        gratings_start = aligned_data[~aligned_data['Stim'].isin([0, np.nan])].index #np.unique(aligned_data['Stim'].dropna()) 
        trial_start_indices = aligned_data[aligned_data['Teleport'] == 1].index.tolist()
        stim_type = aligned_data[~aligned_data['Stim'].isin([0, np.nan])]
        reward_delivery = aligned_data[aligned_data['Reward'] == 1].index.tolist()
        condition = (gratings_start > trial_start_indices[0]) & (gratings_start < trial_start_indices[-1]) # all gratings after the first trial ends (because scanner starts during), and all before the end of the last trial
        gratings_start = gratings_start[condition]
        stim_type = stim_type[condition]
        log_file = os.path.join(self.behavior_data_path,'TrialLogging.yaml')
        trial_types = np.array(self.extract_trial_type(log_file,'trialTypeLabel')[1:-1])
        n_gratings = round(len(gratings_start)/len(trial_start_indices))
        # group grating onsets
        grating_onsets_dict = {f'gr_{i+1}': [] for i in range(n_gratings)}
        for i, idx in enumerate(gratings_start):
            grating_num = (i % n_gratings) + 1
            grating_onsets_dict[f'gr_{grating_num}'].append(idx)
        
        pre_frames = 20
        post_frames = 40
        total_frames = pre_frames + post_frames  # 60 fr total
        total_duration = 7.9  # seconds
        frame_duration = total_duration / total_frames  # seconds per frame

        grating_indices = {}
        time_values = aligned_data['Time'].values
        # generalizable across any set of recorded planes
        for grating, onset_indices  in grating_onsets_dict.items():
            expanded = []
            for onset_idx  in onset_indices:
                onset_time  = aligned_data['Time'].iloc[onset_idx]
                target_times = np.array([onset_time + ((i - pre_frames) * frame_duration) for i in range(total_frames)]) # create target time centered around 0 by finding the nearest original timestamps to these target times 
                closest_indices = []
                for i, t in enumerate(target_times):
                    if i < pre_frames:  # for pre-stim
                        valid_indices = np.where(time_values < onset_time)[0]
                        idx = valid_indices[np.argmin(np.abs(time_values[valid_indices] - t))]
                    else:  # for post-stim
                        valid_indices = np.where(time_values >= onset_time)[0]
                        idx = valid_indices[np.argmin(np.abs(time_values[valid_indices] - t))]

                    closest_indices.append(idx)
                
                closest_indices = np.clip(np.array(closest_indices), 0, len(time_values) - 1) # don't exceed boundaries of data acq
                expanded.append(closest_indices)

            grating_indices[grating] = expanded
            
        # same but with reward 
        post_frames = pre_frames
        total_frames = pre_frames + post_frames  # 60 frames total
        total_duration = 4.8  # seconds
        frame_duration = total_duration / total_frames  # seconds per frame
        reward_indices = []
        for idx in reward_delivery:
            reward_time = aligned_data['Time'].iloc[idx]
            target_times = np.array([reward_time + ((i - pre_frames) * frame_duration) for i in range(total_frames)])
            closest_indices = []
            for i, t in enumerate(target_times):
                if i < pre_frames: 
                    valid_indices = np.where(time_values < reward_time)[0]
                    idx = valid_indices[np.argmin(np.abs(time_values[valid_indices] - t))]
                else: 
                    valid_indices = np.where(time_values >= reward_time)[0]
                    idx = valid_indices[np.argmin(np.abs(time_values[valid_indices] - t))]        

                closest_indices.append(idx)

            closest_indices = np.clip(np.array(closest_indices), 0, len(time_values) - 1)
            reward_indices.append(closest_indices)
        
        # define trial types
        n_trials = int(len(gratings_start)/n_gratings) # list(range(len(grating_onsets_dict['gr_4'])))
        n_trial_types = trial_types.max()+1
        tt_mat = np.zeros((n_trials, n_trial_types), dtype=int)
        tt_mat[np.arange(n_trials), trial_types[:n_trials]] = 1
        
        # unpred_trials['gr_2'] = np.sort(np.concatenate((np.where(tt_mat[:,3])[0],np.where(tt_mat[:,2])[0])))        
        # unpred_trials['gr_4'] = np.sort(np.concatenate((np.where(tt_mat[:,5])[0],np.where(tt_mat[:,2])[0])))
        all_stimuli = set(np.unique(aligned_data['Stim'].dropna()))
        expected_stim_set = {2, 4}  # always expected
        unpred_stim_set = all_stimuli - expected_stim_set
        unpred_trials = {f'gr_{i+1}': [] for i in range(n_gratings)}
        pred_trials = []

        for gr, indices in grating_onsets_dict.items():
            for i, idx in enumerate(indices):
                if stim_type['Stim'][idx] in unpred_stim_set:
                    unpred_trials[gr].append(i)  # Use dynamic gr key
                else:
                    pred_trials.append(i)
        
        return n_gratings,aligned_data, grating_indices, reward_indices, n_trials, tt_mat, trial_start_indices, pred_trials, unpred_trials
    
class Suite2pLoader:
    def process_suite2p_data(self, suite2ppath,neuropil_factor):
        ops = np.load(os.path.join(suite2ppath, f'plane0','ops.npy'), allow_pickle=True).item()
        dF_F = []
        dF_F_red = []
        all_cells_indices = []
        n_planes = ops['nplanes']

        for plane_num in range(n_planes):
            plane_path = os.path.join(suite2ppath, f'plane{plane_num}')
            F = np.load(os.path.join(plane_path, 'F.npy'))
            Fneu = np.load(os.path.join(plane_path, 'Fneu.npy'))
            iscell = np.load(os.path.join(plane_path, 'iscell.npy'))
            cells_indices = np.where(iscell[:, 0])[0]
            TC = (F[cells_indices,:]) - (neuropil_factor * (Fneu[cells_indices,:]))
            # TC = (F) - (neuropil_factor * (Fneu))
            # TC=F
            TC_smoothed = gaussian_filter1d(TC, sigma=0.7, axis=1)
            baseline = np.percentile(TC_smoothed,50,axis=1)
            dF = TC_smoothed - baseline[:, None]
            TC_dff = dF.T / baseline
            TC_dff_Zscore = (TC_dff - np.nanmedian(TC_dff, axis=0)) / np.nanstd(TC_dff, axis=0)
            dF_F.append(TC_dff_Zscore.T)
            all_cells_indices.append(cells_indices)

            try:
                F_red = np.load(os.path.join(plane_path, 'F_chan2.npy'))
                TC_red = F_red[cells_indices,:]
                TC_red_smoothed = gaussian_filter1d(TC_red, sigma=1, axis=1)
                baseline_red = np.percentile(TC_red_smoothed,50,axis=1)
                dF_red = TC_red_smoothed - baseline_red[:, None]
                TC_red_dff = dF_red.T / baseline_red
                absolute_fluorescence_red = TC_red_smoothed
                dF_F_red.append(absolute_fluorescence_red)
            except FileNotFoundError:
                pass

        max_columns = max(array.shape[1] for array in dF_F) # max num of columns
        for i in range(len(dF_F)): #pad arrays with fewer columns than the maximum with NaN values
            if dF_F[i].shape[1] < max_columns:
                padding_size = max_columns - dF_F[i].shape[1]
                dF_F[i] = np.hstack([dF_F[i], np.full((dF_F[i].shape[0], padding_size), np.nan)])

        dF_F = np.vstack(dF_F)
        if dF_F_red:
            for i in range(len(dF_F_red)): #pad arrays with fewer columns than the maximum with NaN values
                if dF_F_red[i].shape[1] < max_columns:
                    padding_size = max_columns - dF_F_red[i].shape[1]
                    dF_F_red[i] = np.hstack([dF_F_red[i], np.full((dF_F_red[i].shape[0], padding_size), np.nan)])
            dF_F_red = np.vstack(dF_F_red)
        return ops, dF_F, all_cells_indices, n_planes, iscell, TC, dF_F_red

class DenoiseData():
    def __init__(self, grating_indices):
        self.grating_indices = grating_indices
    
    def baseline_subtraction(self, zscore, zscore_red, trial_start_indices):
        trial_start_frames =15
        for trial_idx in range(len(trial_start_indices) - 1): 
            start_idx = trial_start_indices[trial_idx] 
            end_idx = trial_start_indices[trial_idx + 1] 
            baseline = np.median(zscore[start_idx : start_idx + trial_start_frames], axis=0)
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
                    # Calculate the noise distribution
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
        # Define instance variables
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

    # DATA LOADING AND INITIALIZING VARIABLES
    def preprocessing(self, table, sheet_name, neuropil_factor, ani, tri_perc, suite2ppath, behavior_data_path,basesub ):
        # Create DataLoader instance
        data_loader = DataLoader(behavior_data_path, sheet_name, suite2ppath)
        suite2p_loader = Suite2pLoader()

        if table.iloc[ani][1] == 1:
            matlab_data = data_loader.load_matlab_data()
            self.n_gratings, self.unpred_trials, self.pred_trials, self.trial_start_indices, self.grating_indices, self.position, num_bins, self.position_tunnel, \
                self.n_trials, self.unpred_led, self.unpred_noled, self.catchA, self.catchB, self.speed = data_loader.load_matlab_variables(matlab_data)
            ops, self.dff_Zscore, all_cells_indices, self.n_planes, iscell,self.TC, self.dff_Zscore_red = suite2p_loader.process_suite2p_data(suite2ppath, neuropil_factor)
            print('matlab done')
        else:
            analog_time, analog_data, fs_analog = data_loader.load_analog_data()
            digital_time, digital_data, fs_digital = data_loader.load_digital_data()
            ops, self.dff_Zscore, all_cells_indices, self.n_planes, iscell,self.TC, self.dff_Zscore_red = suite2p_loader.process_suite2p_data(suite2ppath, neuropil_factor)
            encoder_csv_data, vr_csv_data, events_csv_data, counter_csv_data, quadsync_csv_data, self.log_file, pupil_data = data_loader.load_csv_data()
            downsampled_data, all_pupil_data = data_loader.align_events(digital_data, counter_csv_data, events_csv_data, quadsync_csv_data, vr_csv_data, encoder_csv_data,pupil_data, analog_data, fs_analog, fs_digital, self.n_planes)
            self.n_gratings,self.aligned_data, self.grating_indices, self.reward_indices, self.n_trials, self.trial_types, self.trial_start_indices, self.pred_trials, self.unpred_trials = data_loader.align_data(downsampled_data, self.dff_Zscore)
            print('bonsai done')

        self.denoiser = DenoiseData(self.grating_indices)
        # if self.unpred_trials:
        #     stability_unpred, self.filter_unpred, true_unpred_indices = self.denoiser.filter_trials(tri_perc, self.basesub_activity,self.basesub_activity_red, self.unpred_trials)
        # else:
        #     self.filter_unpred = []
        # stability_pred, self.filter_pred, true_pred_indices = self.denoiser.filter_trials(tri_perc, self.basesub_activity, self.basesub_activity_red, self.pred_trials)
        if basesub == 1:
            self.dff_Zscore, self.dff_Zscore_red = self.denoiser.baseline_subtraction(self.dff_Zscore, self.dff_Zscore_red, self.trial_start_indices)

        for gr, _ in self.grating_indices.items():
            self.activity[gr] = self.dff_Zscore[:,self.grating_indices[gr]]

def main(ani, table,sheet_name, neuropil_factor, tri_perc, basesub = 0):
    suite2ppath = table.iloc[ani][5]  
    behavior_data_path = table.iloc[ani][2]
    
    data_processor = DataProcessor()
    data_processor.preprocessing(table, sheet_name, neuropil_factor, ani, tri_perc, suite2ppath, behavior_data_path, basesub)

    return ani, data_processor

if __name__ == "__main__":
    main()
