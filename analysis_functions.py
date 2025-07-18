import numpy as np
from scipy.signal import savgol_filter
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import warnings
warnings.filterwarnings("ignore")


# FUNCTIONS
def sliding_window_projection(data, pca, window_size=5, step_size=1):
    """
    Projects the sliding window averages of data onto the PCA space.

    Parameters:
    - data: (neurons x trials) matrix representing neural activity.
    - pca: Pre-fitted PCA object for projection.
    - window_size: Number of trials in each window.
    - step_size: How much the window shifts Ater each step.

    Returns:
    - projected_points: List of projected PCA points for each window.
    """
    n_neurons, n_trials = data.shape
    projected_points = []
    
    # Apply sliding window across trials
    for start in range(0, n_trials - window_size + 1, step_size):
        window_data = data[:, start:start + window_size]  # Extract the sliding window
        window_mean = np.mean(window_data, axis=1)  # Average across trials in the window
        projected_point = pca.transform(window_mean.reshape(1, -1))  # (1 x neurons)
        projected_points.append(projected_point[0])  # Append the PCA result
    
    return np.array(projected_points)

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

def linear_svm(X_A, X_B, X_early, X_late, kfolds=10, perc_pca=0.8):
    # dataine A and B data
    X = np.concatenate([X_A, X_B], axis=0)
    y = np.zeros(X.shape[0])
    y[X_A.shape[0]:] = 1  # Assign labels (0 for A, 1 for B)
    
    acc_list = []
    confusion_matrices = []
    early_misclass = []
    late_misclass = []
    skf = StratifiedKFold(n_splits=kfolds, shuffle=True)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    for train_idx, test_idx in skf.split(X, y):
    # for _ in range(kfolds):
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pca = PCA(n_components=perc_pca)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        svm = LinearSVC(penalty='l2', dual=False, C=0.001, class_weight='balanced', max_iter=5000)
        svm.fit(X_train_pca, y_train)
        y_pred = svm.predict(X_test_pca)
        
        svm = LinearSVC(dual=False, class_weight='balanced', max_iter=5000)
        grid_search = GridSearchCV(svm, param_grid, cv=6, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_pca, y_train)
        svm = grid_search.best_estimator_  # Best model
        # print(f"Best C: {grid_search.best_params_['C']}")
        y_pred = svm.predict(X_test_pca)

        acc_list.append(accuracy_score(y_pred, y_test))
        
        cm = confusion_matrix(y_test, y_pred, normalize = 'true')
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        confusion_matrices.append(cm)

        early_pred = svm.predict(pca.transform(X_early))
        late_pred = svm.predict(pca.transform(X_late))
        early_misclass.append(np.sum(early_pred) / X_early.shape[0])
        late_misclass.append(np.sum(late_pred) / X_late.shape[0])

    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)

    return acc_list, avg_confusion_matrix, early_misclass, late_misclass

def find_significant_neurons(act_tri, trial_subset, threshold, prestim_frames, poststim_frames):
    significant_neurons = []
    pvalb2 = []
    thresh = []
    alpha =0.05
    for neuron in range(act_tri.shape[0]):
        pre_stim = np.mean(act_tri[neuron, trial_subset, prestim_frames], axis=1)
        post_stim = np.mean(act_tri[neuron, trial_subset, poststim_frames], axis=1)
        t_stat, pval = ttest_rel(post_stim, pre_stim, alternative='greater')
        pvalb2.append(pval)
        mean_diff_early = np.mean(post_stim) - np.mean(pre_stim)
        thresh.append(int(mean_diff_early > threshold and np.mean(post_stim) > threshold))
        if pval < alpha and np.mean(post_stim) > 0.3:
            significant_neurons.append(neuron)

    return significant_neurons, pvalb2, thresh

def compute_si(blo2activity_stim1, blo2activity_stim2, pooled_neurons, poststim_frames):
    resps1 = np.mean(blo2activity_stim1[pooled_neurons,:,poststim_frames],axis=2)
    resps2 = np.mean(blo2activity_stim2[pooled_neurons,:,poststim_frames],axis=2)
    mean1 = np.mean(resps1, axis=1)
    mean2 = np.mean(resps2, axis=1)
    return (mean1 - mean2) / (mean1 + mean2)

def train_svm_multiclass(X, y, kfolds=5):
    acc = []
    acc_shuff = []
    confusion_matrices = []

    skf = StratifiedKFold(n_splits=kfolds, shuffle=True)
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pca = PCA(n_components=0.9)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        svm = SVC(kernel='linear', C=0.1) 
        svm.fit(X_train_pca, y_train)
        y_pred = svm.predict(X_test_pca)
        acc.append(accuracy_score(y_test, y_pred))
        
        svm_shuff = SVC(kernel='linear', C=0.01) 
        y_train_shuffled = np.random.permutation(y_train)
        svm_shuff.fit(X_train_pca,y_train_shuffled)
        yshuff_pred = svm_shuff.predict(X_test_pca)
        acc_shuff.append(accuracy_score(y_test, yshuff_pred))
        cm = confusion_matrix(y_test, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        confusion_matrices.append(cm)

    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
    return np.mean(acc), np.mean(acc_shuff), avg_confusion_matrix

def plot_trajectory(avg_projected, cmap, ax,pcnums):
    pc0, pc1, pc2 = pcnums  
    for i in range(len(avg_projected) - 1):
        if i == 0:
            ax.scatter(avg_projected[0, pc0], avg_projected[0, pc1], avg_projected[0, pc2], 
                       color=cmap, marker='o', facecolors='none', s=100, linewidths=2)
        elif i == len(avg_projected) - 2:
            ax.scatter(avg_projected[-1, pc0], avg_projected[-1, pc1], avg_projected[-1, pc2], 
                       color=cmap, marker='o', s=100)
        ax.plot(avg_projected[i:i + 2, pc0], avg_projected[i:i + 2, pc1], avg_projected[i:i + 2, pc2], 
                color=cmap, linewidth=2)
        
def get_optimal_bins(data):
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * iqr / (len(data) ** (1/3))
    data_range = max(data) - min(data)
    if bin_width == 0 or np.isnan(bin_width):
        bins = 20  #default
    else:
        bins = max(10, min(50, int(data_range / bin_width)))
    return bins