import numpy as np
import analysis_functions as af

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
    af.plot_shaded_error(axes, range(t_frames), np.mean(np.stack([data[ani][behv][early_unexp_idx[ani]] for ani in animals]), axis = 1),color = '#E57373', alpha=0.2,label=f'X Trials {early_tri.start+1}-{early_tri.stop}')
    af.plot_shaded_error(axes, range(t_frames), np.mean(np.stack([data[ani][behv][late_unexp_idx[ani]] for ani in animals]), axis = 1),color = '#E57373', alpha=0.2,label=f'X Trials {late_tri.start+1}-{late_tri.stop}', style='dash')

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

    if stim_type == 'rw':
        for ani in data:
            early_unexp_idx[ani] = []
            late_unexp_idx[ani] = []
            exp_idx[ani] = []
            for trial in data[ani]['unpred_trials']['gr_2'][early_tri]:
                early_unexp_idx[ani].append(range(data[ani]['reward_indices'][trial]-30,data[ani]['reward_indices'][trial]+31))
            for trial in data[ani]['unpred_trials']['gr_2'][late_tri]:
                late_unexp_idx[ani].append(range(data[ani]['reward_indices'][trial]-30,data[ani]['reward_indices'][trial]+31))
            for trial in data[ani]['unpred_trials']['gr_2'][exp_tri]:
                exp_idx[ani].append(range(data[ani]['reward_indices'][trial-1]-30,data[ani]['reward_indices'][trial-1]+31))        

    t_frames = np.array(exp_idx[ani]).shape[1]
    af.plot_shaded_error(axes, range(t_frames), np.mean(np.stack([data[ani][behv][exp_idx[ani]] for ani in animals]), axis = 1),color = '#5C9BD5', alpha=0.2,label=f'B Block 1 (X Trials {exp_tri.start+1}-{exp_tri.stop})')
    af.plot_shaded_error(axes, range(t_frames), np.mean(np.stack([data[ani][behv][early_unexp_idx[ani]] for ani in animals]), axis = 1),color = '#E57373', alpha=0.2,label=f'X Trials {early_tri.start+1}-{early_tri.stop}')
    af.plot_shaded_error(axes, range(t_frames), np.mean(np.stack([data[ani][behv][late_unexp_idx[ani]] for ani in animals]), axis = 1),color = '#E57373', alpha=0.2,label=f'X Trials {late_tri.start+1}-{late_tri.stop}', style='dash')

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

            early_unexp_data = np.mean(np.stack(data[ani][behv][early_unexp_trials]), axis=0)
            late_unexp_data = np.mean(np.stack(data[ani][behv][late_unexp_trials]), axis=0)
            exp_data = np.mean(np.stack(data[ani][behv][exp_trials]), axis=0)

            early_unexp_data_list.append(early_unexp_data)   # list containing averaged data by trials in all animal
            late_unexp_data_list.append(late_unexp_data)
            exp_data_list.append(exp_data)

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
