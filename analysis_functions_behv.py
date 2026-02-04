import numpy as np
import analysis_functions as af
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# Bootstrapping (sarah suggested)
def bootstrap(data, animals, axes, all_axes, behv = 'pupil', stim_type = 'grat', n_boot=10000, early_tri=slice(0,6),late_tri=slice(6,12)):   # stim_type = 'grat_1' 'grat_2' or 'rw'

    animals = animals
    # bootstrap randomly sample trials for exp 
    exp_distribution = []
    for b in range(n_boot):
        exp_data_list = []
        for ani in animals:
            exp_trials = []
            if stim_type == 'grat_1':
                exp_trials = np.array(data[ani]['grating_indices']['gr_1'])[data[ani]['pred_trials']]
            if stim_type == 'grat_2':
                exp_trials = np.array(data[ani]['grating_indices']['gr_2'])[data[ani]['pred_trials']]        
            if stim_type == 'rw':
                reward_indices = np.array(data[ani]['reward_indices'])
                exp_trials = np.array([np.arange(idx - 30, idx + 31) for idx in reward_indices[data[ani]['pred_trials']]])
            btr_exp_trials = np.random.choice(np.arange(0,len(exp_trials)), (early_tri.stop - early_tri.start), replace=True)   # bootstrap take 6 tri from block1 
            exp_data = []
            for t in btr_exp_trials:
                exp_data.append(data[ani][behv][exp_trials[t,:]])
            btr_exp_data = np.mean(np.stack(np.array(exp_data)), axis=0)
            exp_data_list.append(btr_exp_data)
        avg_exp_data = np.mean(np.array(exp_data_list),axis=0)
        exp_distribution.append(avg_exp_data)
    exp_distribution = np.array(exp_distribution)
    def mean_ci(data):
        mean = np.mean(data, axis=0)
        lower = np.percentile(data, 2.5, axis=0)
        upper = np.percentile(data, 97.5, axis=0)
        return mean, lower, upper
    exp_mean, exp_lower, exp_upper = mean_ci(exp_distribution)
    # plot exp B
    time = np.arange(len(exp_mean))
    axes.plot(time, exp_mean, color='#5C9BD5', label='B Block 1')
    axes.fill_between(time, exp_lower, exp_upper, color='#5C9BD5', alpha=0.2)

    # make avg plot for unexp early and late
    early_unexp_idx = {}
    late_unexp_idx = {}
    if stim_type == 'grat_1':
        for ani in data:
            early_unexp_idx[ani] = []
            late_unexp_idx[ani] = []
            for trial in data[ani]['unpred_trials']['gr_2'][early_tri]:
                early_unexp_idx[ani].append(data[ani]['grating_indices']['gr_1'][trial])
            for trial in data[ani]['unpred_trials']['gr_2'][late_tri]:
                late_unexp_idx[ani].append(data[ani]['grating_indices']['gr_1'][trial])
    if stim_type == 'grat_2':
        for ani in data:
            early_unexp_idx[ani] = []
            late_unexp_idx[ani] = []
            for trial in data[ani]['unpred_trials']['gr_2'][early_tri]:
                early_unexp_idx[ani].append(data[ani]['grating_indices']['gr_2'][trial])
            for trial in data[ani]['unpred_trials']['gr_2'][late_tri]:
                late_unexp_idx[ani].append(data[ani]['grating_indices']['gr_2'][trial])
    if stim_type == 'rw':
        for ani in data:
            early_unexp_idx[ani] = []
            late_unexp_idx[ani] = []
            for trial in data[ani]['unpred_trials']['gr_2'][early_tri]:
                early_unexp_idx[ani].append(range(data[ani]['reward_indices'][trial]-30,data[ani]['reward_indices'][trial]+31))
            for trial in data[ani]['unpred_trials']['gr_2'][late_tri]:
                late_unexp_idx[ani].append(range(data[ani]['reward_indices'][trial]-30,data[ani]['reward_indices'][trial]+31))
    t_frames = np.array(early_unexp_idx[ani]).shape[1]
    af.plot_shaded_error(axes, range(t_frames), np.nanmean(np.stack([data[ani][behv][early_unexp_idx[ani]] for ani in animals]), axis = 1),color = '#E57373', alpha=0.2,label=f'X Trials {early_tri.start+1}-{early_tri.stop}')
    af.plot_shaded_error(axes, range(t_frames), np.nanmean(np.stack([data[ani][behv][late_unexp_idx[ani]] for ani in animals]), axis = 1),color = '#E57373', alpha=0.2,label=f'X Trials {late_tri.start+1}-{late_tri.stop}', style='dash')

    if stim_type == 'grat_1':
        axes.set_xlabel('Time (s)', fontsize=13)
        axes.set_title(behv + ' around grating 1')
        # axes.axvspan(19, 34, color='gray', alpha=0.15)
        if behv == 'pupil':
            axes.set_ylabel('z-pupil diameter')
            axes.set_ylim([-2,2])
        if behv == 'speed':
            axes.set_ylabel('speed - baseline (m/s)')
            axes.set_ylim([0,0.8])
        if behv == 'lick_rate':
            axes.set_ylabel('lick rate (lick/s)')
        axes.set_xticks([11.5, 19, 37])
        axes.set_xticklabels(['-1', '0', '2.4'])
        axes.plot([19, 19], axes.get_ylim(), color='black', linewidth=2.2, linestyle='-')
        axes.axvline(x=19, color='black', linewidth=2.2, linestyle='-')

    if stim_type == 'grat_2':
        axes.set_xlabel('Time (s)', fontsize=13)
        axes.set_title(behv + ' around grating 2')
        # axes.axvspan(19, 34, color='gray', alpha=0.15)
        if behv == 'pupil':
            axes.set_ylabel('z-pupil diameter')
            axes.set_ylim([-2,2])            
        if behv == 'speed':
            axes.set_ylabel('speed - baseline (m/s)')
            axes.set_ylim([0,0.8])
        if behv == 'lick_rate':
            axes.set_ylabel('lick rate (lick/s)')
        axes.set_xticks([11.5, 19, 37])
        axes.set_xticklabels(['-1', '0', '2.4'])
        axes.plot([19, 19], axes.get_ylim(), color='black', linewidth=2.2, linestyle='-')
        axes.axvline(x=19, color='black', linewidth=2.2, linestyle='-')

    if stim_type == 'rw':
        axes.set_xlabel('Time (s)', fontsize=13)
        if axes is all_axes[0]:
            axes.set_title(behv + ' around reward (ctl)')
        if axes is all_axes[1]:
            axes.set_title(behv + ' around reward (tst)')
        if behv == 'pupil':
            axes.set_ylabel('z-pupil diameter')
            axes.set_ylim([-1.5,1])
        if behv == 'speed':
            axes.set_ylabel('speed - baseline (m/s)')
            axes.set_ylim([0,0.5])
        if behv == 'lick_rate':
            axes.set_ylabel('lick rate (lick/s)')
        axes.set_xticks([22.5, 30, 48])
        axes.set_xticklabels(['-1', '0', '2.4'])
        axes.plot([30, 30], axes.get_ylim(), color='black', linewidth=2.2, linestyle='-')
        axes.axvline(x=30, color='black', linewidth=2.2, linestyle='-')

    if axes is all_axes[0]:
        axes.legend().remove()
    else:
        axes.legend()

    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)    

def avg_plot(data, animals, axes, all_axes, behv = 'pupil', stim_type = 'grat_1', early_tri = slice(0,6), late_tri = slice(6,12), exp_tri = slice(0,16)):
    
    animals = animals
    n_animals = len(animals)
    early_unexp_idx = {}
    late_unexp_idx = {}
    exp_idx = {}

    if stim_type == 'grat_1':
        for ani in data:
            early_unexp_idx[ani] = []
            late_unexp_idx[ani] = []
            exp_idx[ani] = []
            for trial in data[ani]['unpred_trials']['gr_2'][early_tri]:
                early_unexp_idx[ani].append(data[ani]['grating_indices']['gr_1'][trial])
            for trial in data[ani]['unpred_trials']['gr_2'][late_tri]:
                late_unexp_idx[ani].append(data[ani]['grating_indices']['gr_1'][trial])
            for trial in data[ani]['unpred_trials']['gr_2'][exp_tri]:
                exp_idx[ani].append(data[ani]['grating_indices']['gr_1'][trial-1])

    if stim_type == 'grat_2':
        for ani in data:
            early_unexp_idx[ani] = []
            late_unexp_idx[ani] = []
            exp_idx[ani] = []
            for trial in data[ani]['unpred_trials']['gr_2'][early_tri]:
                early_unexp_idx[ani].append(data[ani]['grating_indices']['gr_2'][trial])
            for trial in data[ani]['unpred_trials']['gr_2'][late_tri]:
                late_unexp_idx[ani].append(data[ani]['grating_indices']['gr_2'][trial])
            for trial in data[ani]['unpred_trials']['gr_2'][exp_tri]:
                exp_idx[ani].append(data[ani]['grating_indices']['gr_2'][trial-1])

    # if stim_type == 'rw':
    #     for ani in data:
    #         early_unexp_idx[ani] = []
    #         late_unexp_idx[ani] = []
    #         exp_idx[ani] = []
    #         for trial in data[ani]['unpred_trials']['gr_2'][early_tri]:
    #             early_unexp_idx[ani].append(range(data[ani]['reward_indices'][trial]-30,data[ani]['reward_indices'][trial]+31))
    #         for trial in data[ani]['unpred_trials']['gr_2'][late_tri]:
    #             late_unexp_idx[ani].append(range(data[ani]['reward_indices'][trial]-30,data[ani]['reward_indices'][trial]+31))
    #         for trial in data[ani]['unpred_trials']['gr_2'][exp_tri]:
    #             exp_idx[ani].append(range(data[ani]['reward_indices'][trial-1]-30,data[ani]['reward_indices'][trial-1]+31)) 

    if stim_type == 'rw':
        for ani in data:
            early_unexp_idx[ani] = []
            late_unexp_idx[ani] = []
            exp_idx[ani] = []
            for trial in data[ani]['unpred_trials']['gr_2'][early_tri]:
                early_unexp_idx[ani].append(data[ani]['reward_indices'][trial])
            for trial in data[ani]['unpred_trials']['gr_2'][late_tri]:
                late_unexp_idx[ani].append(data[ani]['reward_indices'][trial])
            for trial in data[ani]['unpred_trials']['gr_2'][exp_tri]:
                exp_idx[ani].append(data[ani]['reward_indices'][trial-1])         

    t_frames = np.array(exp_idx[ani]).shape[1]
    plot_shaded_error(axes, range(t_frames), np.nanmean(np.stack([data[ani][behv][exp_idx[ani]] for ani in animals]), axis = 1),color = '#5C9BD5', alpha=0.2,label=f'B Block 1 (X Trials {exp_tri.start+1}-{exp_tri.stop})')
    # plot_shaded_error(axes, range(t_frames), np.nanmean(np.stack([data[ani][behv][early_unexp_idx[ani]] for ani in animals]), axis = 1),color = '#E57373', alpha=0.2,label=f'X Trials {early_tri.start+1}-{early_tri.stop}')
    # plot_shaded_error(axes, range(t_frames), np.nanmean(np.stack([data[ani][behv][late_unexp_idx[ani]] for ani in animals]), axis = 1),color = '#E57373', alpha=0.2,label=f'X Trials {late_tri.start+1}-{late_tri.stop}', style='dash')

    if stim_type == 'grat_1':
        axes.set_xlabel('Time (s)', fontsize=13)
        axes.set_title(behv + ' around grating 1')
        if behv == 'pupil':
            axes.set_ylabel('z-pupil diameter')
            axes.set_ylim([-2,2])
        if behv == 'speed':
            axes.set_ylabel('speed - baseline (m/s)')
            axes.set_ylim([0,0.8])
        if behv == 'lick_rate':
            axes.set_ylabel('lick rate (lick/s)')
        axes.set_xticks([11.5, 19, 37])
        axes.set_xticklabels(['-1', '0', '2.4'])
        axes.plot([19, 19], axes.get_ylim(), color='black', linewidth=2.2, linestyle='-')
        axes.axvline(x=19, color='black', linewidth=2.2, linestyle='-')


    if stim_type == 'grat_2':
        axes.set_xlabel('Time (s)', fontsize=13)
        axes.set_title(behv + ' around grating 2')
        if behv == 'pupil':
            axes.set_ylabel('z-pupil diameter')
            axes.set_ylim([-2,2])            
        if behv == 'speed':
            axes.set_ylabel('speed - baseline (m/s)')
            axes.set_ylim([0,0.8])
        if behv == 'lick_rate':
            axes.set_ylabel('lick rate (lick/s)')
        axes.set_xticks([11.5, 19, 37])
        axes.set_xticklabels(['-1', '0', '2.4'])
        axes.plot([19, 19], axes.get_ylim(), color='black', linewidth=2.2, linestyle='-')
        axes.axvline(x=19, color='black', linewidth=2.2, linestyle='-')


    if stim_type == 'rw':
        axes.set_xlabel('Time (s)', fontsize=13)
        if axes is all_axes[0]:
            axes.set_title(behv + ' around reward (ctl)')
        if axes is all_axes[1]:
            axes.set_title(behv + ' around reward (tst)')
        if behv == 'pupil':
            axes.set_ylabel('z-pupil diameter')
            axes.set_ylim([-1.5,1])
        if behv == 'speed':
            axes.set_ylabel('speed - baseline (m/s)')
            axes.set_ylim([0,0.5])
        if behv == 'lick_rate':
            axes.set_ylabel('lick rate (lick/s)')
        # axes.set_xticks([22.5, 30, 48])
        # axes.set_xticklabels(['-1', '0', '2.4'])
        axes.plot([20, 20], axes.get_ylim(), color='black', linewidth=2.2, linestyle='-')  
        # axes.axvline(x=20, color='black', linewidth=2.2, linestyle='-')


    if axes is all_axes[0]:
        axes.legend().remove()
    else:
        axes.legend()

    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)    


# normal bootstrapping
def bootstrap_normal(data, animals, axes, behv = 'pupil', stim_type = 'grat', n_boot=10000, early_tri=slice(0,6),late_tri=slice(6,12)):   # stim_type = 'grat_1' 'grat_2' or 'rw'
# normal bootstraping (in each bootstrap, resample N animals, and calculate the value in those animals without any operation)
    animals = animals
    n_animals = len(animals)
    early_unexp_distribution = []
    late_unexp_distribution = []
    exp_distribution = []

    for b in range(n_boot):
        sampled_animals = np.random.choice(animals, n_animals, replace=True)
        early_unexp_data_list = []
        late_unexp_data_list = []
        exp_data_list = []
        for ani in sampled_animals:
            early_unexp_trials = []
            late_unexp_trials = []
            exp_trials = []
            if stim_type == 'grat_1':
                early_unexp_trials = np.array(data[ani]['grating_indices']['gr_1'])[data[ani]['unpred_trials']['gr_2'][early_tri]]
                late_unexp_trials = np.array(data[ani]['grating_indices']['gr_1'])[data[ani]['unpred_trials']['gr_2'][late_tri]]
                exp_trials = np.array(data[ani]['grating_indices']['gr_1'])[data[ani]['pred_trials']]
            if stim_type == 'grat_2':
                early_unexp_trials = np.array(data[ani]['grating_indices']['gr_2'])[data[ani]['unpred_trials']['gr_2'][early_tri]]
                late_unexp_trials = np.array(data[ani]['grating_indices']['gr_2'])[data[ani]['unpred_trials']['gr_2'][late_tri]]
                exp_trials = np.array(data[ani]['grating_indices']['gr_2'])[data[ani]['pred_trials']]        
            if stim_type == 'rw':
                reward_indices = np.array(data[ani]['reward_indices'])
                early_unexp_trials = np.array([np.arange(idx - 30, idx + 31) for idx in reward_indices[data[ani]['unpred_trials']['gr_2']]])[early_tri,:]
                late_unexp_trials = np.array([np.arange(idx - 30, idx + 31) for idx in reward_indices[data[ani]['unpred_trials']['gr_2']]])[late_tri,:]
                exp_trials = np.array([np.arange(idx - 30, idx + 31) for idx in reward_indices[data[ani]['pred_trials']]])[slice(early_tri.start,late_tri.stop),:]

            early_unexp_data = np.nanmean(np.stack(data[ani][behv][early_unexp_trials]), axis=0)
            late_unexp_data = np.nanmean(np.stack(data[ani][behv][late_unexp_trials]), axis=0)
            exp_data = np.nanmean(np.stack(data[ani][behv][exp_trials]), axis=0)

            early_unexp_data_list.append(early_unexp_data)   # list containing averaged data by trials in all animal
            late_unexp_data_list.append(late_unexp_data)
            exp_data_list.append(exp_data)

        avg_early_unexp_data = np.nanmean(np.array(early_unexp_data_list),axis=0) # average across animals and get one data trace for this bootstrap
        avg_late_unexp_data = np.nanmean(np.array(late_unexp_data_list),axis=0)
        avg_exp_data = np.nanmean(np.array(exp_data_list),axis=0)

        early_unexp_distribution.append(avg_early_unexp_data)
        late_unexp_distribution.append(avg_late_unexp_data)
        exp_distribution.append(avg_exp_data)

    # Convert lists to numpy arrays for easier computation
    early_unexp_distribution = np.array(early_unexp_distribution)
    late_unexp_distribution = np.array(late_unexp_distribution)
    exp_distribution = np.array(exp_distribution)

    # Compute mean and 95% confidence intervals (2.5th and 97.5th percentiles)
    def mean_ci(data):
        mean = np.nanmean(data, axis=0)
        lower = np.percentile(data, 2.5, axis=0)
        upper = np.percentile(data, 97.5, axis=0)
        return mean, lower, upper

    early_mean, early_lower, early_upper = mean_ci(early_unexp_distribution)
    late_mean, late_lower, late_upper = mean_ci(late_unexp_distribution)
    exp_mean, exp_lower, exp_upper = mean_ci(exp_distribution)

    # Plot
    time = np.arange(len(early_mean))
    # Expected B trial
    axes.plot(time, exp_mean, color='#5C9BD5', label='Expected B trial')
    axes.fill_between(time, exp_lower, exp_upper, color='#5C9BD5', alpha=0.2)
    # Unexpected X trial
    axes.plot(time, early_mean, color='#E57373', label='Unexpected X trial')
    axes.fill_between(time, early_lower, early_upper, color='#E57373', alpha=0.2)
    # # Expected X trial
    axes.plot(time, late_mean, color='#E57373', linestyle='--', label='Expected X trial')
    axes.fill_between(time, late_lower, late_upper, color='#E57373', alpha=0.2)

    if stim_type == 'grat_1':
        axes.set_xlabel('Time (frames)')
        axes.set_title(behv + ' around grating 1')
        axes.axvspan(19, 34, color='gray', alpha=0.15)
        if behv == 'pupil':
            axes.set_ylabel('z-pupil diameter')
            axes.set_ylim([-2,2])
        if behv == 'speed':
            axes.set_ylabel('speed -baseline (cm/s)')
        if behv == 'lick':
            axes.set_ylabel('lick rate (lick/s)')
    if stim_type == 'grat_2':
        axes.set_xlabel('Time (frames)')
        axes.set_title(behv + ' around grating 2')
        axes.axvspan(19, 34, color='gray', alpha=0.15)
        if behv == 'pupil':
            axes.set_ylabel('z-pupil diameter')
        if behv == 'speed':
            axes.set_ylabel('speed -baseline (cm/s)')
        if behv == 'lick':
            axes.set_ylabel('lick rate (lick/s)')
    if stim_type == 'rw':
        axes.set_xlabel('Time (frames)')
        axes.set_title(behv + ' around reward')
        axes.axvline(x=30, color='k', linestyle='--')
        if behv == 'pupil':
            axes.set_ylabel('z-pupil diameter')
        if behv == 'speed':
            axes.set_ylabel('speed -baseline (cm/s)')
        if behv == 'lick':
            axes.set_ylabel('lick rate (lick/s)')
    axes.legend()


# Hierarchical Bootstrapping
def bootstrap_nested(data, animals, axes, behv = 'pupil', stim_type = 'grat', n_boot=10000, early_tri=slice(0,6),late_tri=slice(6,12)):   # stim_type = 'grat_1' 'grat_2' or 'rw'

    animals = animals
    n_animals = len(animals)
    early_unexp_distribution = []
    late_unexp_distribution = []
    exp_distribution = []

    for b in range(n_boot):
        sampled_animals = np.random.choice(animals, n_animals, replace=True)
        early_unexp_data_list = []
        late_unexp_data_list = []
        exp_data_list = []
        for ani in sampled_animals:
            early_unexp_trials = []
            late_unexp_trials = []
            exp_trials = []

            if stim_type == 'grat_1':
                early_unexp_trials = np.array(data[ani]['grating_indices']['gr_1'])[data[ani]['unpred_trials']['gr_2'][early_tri]]
                late_unexp_trials = np.array(data[ani]['grating_indices']['gr_1'])[data[ani]['unpred_trials']['gr_2'][late_tri]]
                exp_trials = np.array(data[ani]['grating_indices']['gr_1'])[data[ani]['pred_trials']]
            if stim_type == 'grat_2':
                early_unexp_trials = np.array(data[ani]['grating_indices']['gr_2'])[data[ani]['unpred_trials']['gr_2'][early_tri]]
                late_unexp_trials = np.array(data[ani]['grating_indices']['gr_2'])[data[ani]['unpred_trials']['gr_2'][late_tri]]
                exp_trials = np.array(data[ani]['grating_indices']['gr_2'])[data[ani]['pred_trials']]        
            if stim_type == 'rw':
                reward_indices = np.array(data[ani]['reward_indices'])
                early_unexp_trials = np.array([np.arange(idx - 30, idx + 31) for idx in reward_indices[data[ani]['unpred_trials']['gr_2']]])[early_tri,:]
                late_unexp_trials = np.array([np.arange(idx - 30, idx + 31) for idx in reward_indices[data[ani]['unpred_trials']['gr_2']]])[late_tri,:]
                exp_trials = np.array([np.arange(idx - 30, idx + 31) for idx in reward_indices[data[ani]['pred_trials']]])

            btr_early_unexp_trials = np.random.choice(np.arange(0,len(early_unexp_trials)), len(early_unexp_trials), replace=True)
            btr_late_unexp_trials = np.random.choice(np.arange(0,len(late_unexp_trials)), len(late_unexp_trials), replace=True)
            btr_exp_trials = np.random.choice(np.arange(0,len(exp_trials)), len(exp_trials), replace=True)

            early_unexp_data = []
            for t in btr_early_unexp_trials:
                early_unexp_data.append(data[ani][behv][early_unexp_trials[t,:]])
            late_unexp_data = []
            for t in btr_late_unexp_trials:
                late_unexp_data.append(data[ani][behv][late_unexp_trials[t,:]])            
            exp_data = []
            for t in btr_exp_trials:
                exp_data.append(data[ani][behv][exp_trials[t,:]])

            btr_early_unexp_data = np.mean(np.stack(np.array(early_unexp_data)), axis=0)
            btr_late_unexp_data = np.mean(np.stack(np.array(late_unexp_data)), axis=0)
            btr_exp_data = np.mean(np.stack(np.array(exp_data)), axis=0)

            early_unexp_data_list.append(btr_early_unexp_data)   # list containing averaged data by trials in all animal
            late_unexp_data_list.append(btr_late_unexp_data)
            exp_data_list.append(btr_exp_data)

        avg_early_unexp_data = np.mean(np.array(early_unexp_data_list),axis=0) # average across animals and get one data trace for this bootstrap
        avg_late_unexp_data = np.mean(np.array(late_unexp_data_list),axis=0)
        avg_exp_data = np.mean(np.array(exp_data_list),axis=0)

        early_unexp_distribution.append(avg_early_unexp_data)
        late_unexp_distribution.append(avg_late_unexp_data)
        exp_distribution.append(avg_exp_data)

    # Convert lists to numpy arrays for easier computation
    early_unexp_distribution = np.array(early_unexp_distribution)
    late_unexp_distribution = np.array(late_unexp_distribution)
    exp_distribution = np.array(exp_distribution)

    # Compute mean and 95% confidence intervals (2.5th and 97.5th percentiles)
    def mean_ci(data):
        mean = np.mean(data, axis=0)
        lower = np.percentile(data, 2.5, axis=0)
        upper = np.percentile(data, 97.5, axis=0)
        return mean, lower, upper

    early_mean, early_lower, early_upper = mean_ci(early_unexp_distribution)
    late_mean, late_lower, late_upper = mean_ci(late_unexp_distribution)
    exp_mean, exp_lower, exp_upper = mean_ci(exp_distribution)

    # Plot
    time = np.arange(len(early_mean))
    # Expected B trial
    axes.plot(time, exp_mean, color='#5C9BD5', label='Expected B trial')
    axes.fill_between(time, exp_lower, exp_upper, color='#5C9BD5', alpha=0.2)
    # Unexpected X trial
    axes.plot(time, early_mean, color='#E57373', label='Unexpected X trial')
    axes.fill_between(time, early_lower, early_upper, color='#E57373', alpha=0.2)
    # # Expected X trial
    axes.plot(time, late_mean, color='#E57373', linestyle='--', label='Expected X trial')
    axes.fill_between(time, late_lower, late_upper, color='#E57373', alpha=0.2)

    if stim_type == 'grat_1':
        axes.set_xlabel('Time (frames)')
        axes.set_title(behv + ' around grating 1')
        axes.axvline(x=19, color='k', linestyle='--')
        if behv == 'pupil':
            axes.set_ylabel('z-pupil diameter')
        if behv == 'speed':
            axes.set_ylabel('speed -baseline (cm/s)')
        if behv == 'lick':
            axes.set_ylabel('lick rate (lick/s)')
    if stim_type == 'grat_2':
        axes.set_xlabel('Time (frames)')
        axes.set_title(behv + ' around grating 2')
        axes.axvline(x=19, color='k', linestyle='--')
        if behv == 'pupil':
            axes.set_ylabel('z-pupil diameter')
        if behv == 'speed':
            axes.set_ylabel('speed -baseline (cm/s)')
        if behv == 'lick':
            axes.set_ylabel('lick rate (lick/s)')
    if stim_type == 'rw':
        axes.set_xlabel('Time (frames)')
        axes.set_title(behv + ' around reward')
        axes.axvline(x=30, color='k', linestyle='--')
        if behv == 'pupil':
            axes.set_ylabel('z-pupil diameter')
        if behv == 'speed':
            axes.set_ylabel('speed -baseline (cm/s)')
        if behv == 'lick':
            axes.set_ylabel('lick rate (lick/s)')

    axes.legend()

def plot_shaded_error(axes, x_vals, data, color='k', label=None, alpha=0.2, ylim=None, title=None,style=None):
    """Plot mean with shaded error bars (std) on given axes."""
    mean_vals = np.nanmean(data, axis=0)
    std_vals = np.nanstd(data, axis=0) / np.sqrt(len(mean_vals))

    if style == 'smooth':
        window_size = 3
        mean_vals = savgol_filter(mean_vals, window_length=window_size, polyorder=1)
        std_vals = savgol_filter(std_vals, window_length=window_size, polyorder=1)

    linestyle = '--' if style == 'dash' else '-'
    
    axes.plot(x_vals, mean_vals, color=color, label=label, linestyle=linestyle, linewidth=2)
    axes.fill_between(x_vals, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=alpha)

    if ylim:
        axes.set_ylim(ylim)
    if label:
        axes.legend()
    if title:
        axes.set_title(title)

def plot_detected_licks(ani_number,dataset,trial_number):

    """
        Plot lick signal and detected lick onsets for a single animal
        around a specified trial.

        This function extracts the lick signal from the dataset, detects
        lick onsets using a threshold-based method, and visualizes the
        lick trace together with reward times and detected lick events
        for a selected trial window.

        Parameters
        ----------
        ani_number : str or int
            Animal identifier used as a key in the dataset.
        dataset : dict
            Dictionary containing behavioral data (lick signal, trial indices, etc.).
        trial_number : int
            Index of the trial to be plotted.

        Returns
        -------
        None
            Displays a plot of lick signal and detected lick onsets.
    """

    dataset = dataset
    lick_onsets_ctl = {}
    
    fig, axes = plt.subplots(figsize=(10, 5))

    max_valid = 5
    min_interval = 100   # refractory period in samples

    for ani in dataset:
        lick_onsets_ctl[ani] = []

        lick_signal_raw = np.clip(dataset[ani]['lick'], a_min=None, a_max=max_valid)
        lick_signal = gaussian_filter1d(lick_signal_raw, sigma=0.1)  # optional smoothing

        trial_start_indices = dataset[ani]['trial_start_indices']

        # iterate over all trials
        for tri in range(len(trial_start_indices) - 1):
            start = trial_start_indices[tri]
            end = trial_start_indices[tri + 1]
            segment = lick_signal[start:end]

            # lower, upper = np.percentile(segment, [20, 80]) 
            # clipped = segment[(segment > lower) & (segment < upper)]
            # threshold = np.median(clipped) + 5 * np.std(clipped)
            # print(threshold)

            # compute per-trial threshold
            threshold = np.median(segment) + 3 * np.std(segment)

            # find threshold crossings
            above = segment > threshold
            crossings = np.where((~above[:-1]) & (above[1:]))[0] + 1 

            # convert local indices to absolute indices
            if len(crossings) > 0:
                # enforce refractory period (in absolute frame)
                abs_cross = start + crossings
                if len(abs_cross) > 0:
                    filtered = [abs_cross[0]]
                    for idx in abs_cross[1:]:
                        if idx - filtered[-1] >= min_interval:
                            filtered.append(idx)
                    lick_onsets_ctl[ani].extend(filtered)

    # convert to numpy arrays for convenience
    ani = ani_number
    lick_onsets_ctl[ani] = np.array(lick_onsets_ctl[ani])

    reward_indices = dataset[ani]['reward_indices']
    trial_start_indices = dataset[ani]['trial_start_indices']

    tri = trial_number

    start, end = trial_start_indices[tri], trial_start_indices[tri+5]
    lick_signal = dataset[ani]['lick'] 

    axes.plot(lick_signal[start:end])
    axes.set_title(f'{ani}')
    axes.set_ylim([0,1])
    # axes.axhline(0.3,c='red')
    axes.axvline(reward_indices[tri+1]-start,c='red')
    axes.axvline(reward_indices[tri+2]-start,c='red')
    axes.axvline(reward_indices[tri+3]-start,c='red')
    axes.axvline(reward_indices[tri+4]-start,c='red')

    onsets_in_segment = lick_onsets_ctl[ani][
        (lick_onsets_ctl[ani] >= start) & (lick_onsets_ctl[ani] < end)
    ]

    rel_onsets = onsets_in_segment - start

    axes.scatter(
        rel_onsets,
        lick_signal[onsets_in_segment],
        color='red', s=3, zorder=3
    )

def extract_lick_events_by_position(ani, data, trial_type, plot = True):

    """
        This function iterates through all trials of a given animal, extracts lick events,
        and aligns them to spatial position within each trial. Licks are separated into
        expected and unexpected trials based on trial type labels stored in the dataset.

        Parameters
        ----------
        ani : int
            Animal ID.
        data : dict
            Dictionary containing behavioral data for all animals. Each animal entry
        trial_type : str
            Which trials to plot. Must be one of:
            {'Expected', 'Unexpected', 'Both'}.
        plot : bool, optional
            Whether to generate a raster plot of lick positions (default: True).

        Returns
        -------
        aligned_exp : np.ndarray
            Array of shape (N, 2) containing [trial_index, position] for licks
            occurring in expected trials.
        aligned_unexp : np.ndarray
            Array of shape (M, 2) containing [trial_index, position] for licks
            occurring in unexpected trials.

        Notes
        -----
        - Positions are converted to centimeters using: position * 700 / 5
        - Each row in the output arrays corresponds to a single lick event
        - If plot=True, a raster plot of lick positions across trials is shown
        """

    ani_positions = data[ani]['real_position'] * 700/5
    aligned_exp = []
    aligned_unexp = []
    for i, trial_start in enumerate(data[ani]['trial_start_indices']):   # i is the trial number and trial_start is the index
        if i < len(data[ani]['trial_start_indices']) - 1:   
            trial_end = data[ani]['trial_start_indices'][i+1]   
        else:
            trial_end = len(data[ani]['real_position'])
        for num, lick in enumerate(data[ani]['lick_record'][trial_start:trial_end]):   # num is the index in [trial_start:trial_end]
            if lick == 1 and i in data[ani]['unpred_trials']['gr_2']:
                aligned_unexp.append([i,ani_positions[trial_start:trial_end][num]])
            if lick == 1 and i in data[ani]['pred_trials']:
                aligned_exp.append([i,ani_positions[trial_start:trial_end][num]])
    aligned_exp = np.array(aligned_exp)
    aligned_unexp = np.array(aligned_unexp)
    
    if plot == True:
        fig,ax = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
        if trial_type == 'Unexpected':
            ax.scatter(aligned_unexp[:, 1], aligned_unexp[:, 0], s=5, c='#C51B8A', label = 'Unexpected')
        elif trial_type == 'Expected':
            ax.scatter(aligned_exp[:, 1], aligned_exp[:, 0], s=5, c='#00A551', label = 'Expected')
        elif trial_type == 'Both':
            ax.scatter(aligned_unexp[:, 1], aligned_unexp[:, 0], s=5, c='#C51B8A', label = 'Unexpected')
            ax.scatter(aligned_exp[:, 1], aligned_exp[:, 0], s=5, c='#00A551', label = 'Expected')
        else:
            raise ValueError("trial_type must be one of: 'Unexpected', 'Expected', 'Both'")
        ax.axvline(x=4.62*700/5, color='red', linestyle='--', linewidth=1, label='Reward position')
        ax.legend(loc='upper left')
        ax.set_xlim(0, 700)
        ax.set_title('Ani ' + str(ani) + ' ' + trial_type)
        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Trial number')
        plt.tight_layout()
        plt.show()

    return aligned_exp, aligned_unexp