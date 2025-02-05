import data_utils

from scipy.stats import pearsonr, kendalltau, spearmanr
from scipy.spatial.distance import pdist, squareform, cosine, euclidean
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np

def basic_rsa(representations, Y, save_fig=True):

    ordered_indices = np.argsort(Y)[::-1]
    ordered_Y = Y[ordered_indices]
    ordered_representations = representations[ordered_indices]
    
    model_rdm = get_model_rdm(ordered_Y)

    data_rdm = get_rdm(ordered_representations, distance_metric='euclidean')
    
    result = correlate_rdms(model_rdm, data_rdm)

    return result.correlation, result.pvalue

def random_rsa(representations, Y):

    ordered_indices = np.argsort(Y)
    ordered_Y = Y[ordered_indices]
    ordered_representations = representations[ordered_indices]

    random_ground_truth = get_model_rdm(shuffle(Y))

    data_rdm = get_rdm(ordered_representations, distance_metric='euclidean')

    result = correlate_rdms(random_ground_truth, data_rdm)

    return result.correlation, result.pvalue

def correlate_over_groups(rdm_a, rdm_b, second_rdm_group_level_already=False, within_compound_sentence_corrs=False, corr_metric='kendalltau'):

    within_sentence_inds = np.arange(15) % 3 == 0
    
    if within_compound_sentence_corrs and second_rdm_group_level_already:
        rdm_b = rdm_b[within_sentence_inds, :][:, within_sentence_inds]
    
    corrs = []
    for group_i in range(60):
        if second_rdm_group_level_already:
            corr = correlate_rdms(data_utils.select_within_compound_groups(rdm_a, group_i, within_compound_sentence_corrs), get_lower_triangle(rdm_b), correlation=corr_metric, select_lower_triangle=False)
        else:
            corr = correlate_rdms(data_utils.select_within_compound_groups(rdm_a, group_i, within_compound_sentence_corrs), data_utils.select_within_compound_groups(rdm_b, group_i, within_compound_sentence_corrs),
                                      correlation=corr_metric, select_lower_triangle=False)

        corrs.append(corr.correlation)
    return corrs

def correlate_and_average_over_groups(rdm_a, rdm_b, second_rdm_group_level_already=False, within_compound_sentence_corrs=False, corr_metric='kendalltau', keep_corrs=False):

    corrs = correlate_over_groups(rdm_a, rdm_b, second_rdm_group_level_already, within_compound_sentence_corrs, corr_metric)
        
    if keep_corrs:
        return np.mean(corrs), np.std(corrs), corrs
    else:
        return np.mean(corrs), np.std(corrs)

def correlate_over_groups_and_get_row_values(rdm_a, rdm_b, rdm_name, second_rdm_group_level_already=False, include_within_compound_sentence_corrs=True, corr_metric='kendalltau', keep_corrs=False):
    row = {}
    
    corr_val, std = correlate_and_average_over_groups(rdm_a, rdm_b, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)
    row['{}_corr'.format(rdm_name)] = corr_val
    row['{}_std'.format(rdm_name)] = std
    
    if include_within_compound_sentence_corrs:
        if keep_corrs:
            corr_val, std, corrs = correlate_and_average_over_groups(rdm_a, rdm_b, second_rdm_group_level_already, within_compound_sentence_corrs=True, corr_metric=corr_metric, keep_corrs=True)
        else:
            corr_val, std = correlate_and_average_over_groups(rdm_a, rdm_b, second_rdm_group_level_already, within_compound_sentence_corrs=True, corr_metric=corr_metric)
  
        row['{}_corr_within_compound_sentences'.format(rdm_name)] = corr_val
        row['{}_std_within_compound_sentences'.format(rdm_name)] = std
    
    if keep_corrs:
       return row, corrs
    else:
        return row

def sentence_level_rsa(stacked_sentence_cls_tokens, Y, dist='cosine', normalise_features=False):
    if normalise_features:
        stacked_sentence_cls_tokens = normalize(stacked_sentence_cls_tokens, axis=0) # Normalise features across all samples

    inds = np.arange(stacked_sentence_cls_tokens.shape[0])
    sent_1_inds = inds[inds % 2 == 0] # e.g. [0, 1, 2, 3] -> [True, False, True, False]
    cls_tokens_per_pair = np.stack([stacked_sentence_cls_tokens[sent_1_inds], stacked_sentence_cls_tokens[~sent_1_inds]], axis=1)

    sim_per_pair = np.array([1 - cosine(x[0], x[1]) for x in cls_tokens_per_pair])

    ordered_indices = np.argsort(Y)
    ordered_Y = Y[ordered_indices]
    ordered_sim_per_pair = sim_per_pair[ordered_indices]

    ground_truth = get_model_rdm(ordered_Y)

    data_rdm = get_rdm(ordered_sim_per_pair.reshape(-1, 1), distance_metric='euclidean')

    result = correlate_rdms(ground_truth, data_rdm)

    direct_result = pearsonr(sim_per_pair, Y)

    return result.correlation, result.pvalue, direct_result[0], direct_result[1]

def get_rdm(reps, distance_metric='euclidean'):
    return squareform(pdist(normalize(reps, axis=0), metric=distance_metric))

def get_model_rdm(Y):
    return get_rdm(Y.reshape(-1, 1), distance_metric='euclidean')

def plot_mtx(mtx, title, figsize=(8,6)):
    plt.figure(figsize=figsize)
    plt.imshow(mtx, interpolation='nearest', cmap='Spectral_r')
    plt.title(title)
    plt.colorbar()

def get_lower_triangle(rdm):
    return rdm[np.where(np.triu(np.ones(rdm.shape)) == 0)]

def correlate_rdms(rdm_a, rdm_b, correlation="spearmanr", select_lower_triangle=True):
    corr_dict = {'spearmanr': spearmanr, 'kendalltau': kendalltau, 'pearsonr': pearsonr}
    corr_func = corr_dict[correlation]
    
    if select_lower_triangle:
        rdm_a = get_lower_triangle(rdm_a)
        rdm_b = get_lower_triangle(rdm_b)
    
    to_keep_inds = np.argwhere(~np.isnan(rdm_a) & ~np.isnan(rdm_b)).reshape(-1)
    new_rdm_a = rdm_a[to_keep_inds]
    new_rdm_b = rdm_b[to_keep_inds]
    
    return corr_func(np.array(new_rdm_a).reshape(-1, 1), np.array(new_rdm_b).reshape(-1, 1))