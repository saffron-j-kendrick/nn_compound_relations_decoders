import argparse

from sklearn.utils import shuffle
from sklearn import linear_model
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

import data_utils
import model_utils
import representations

def linear_regression(representations, scores):
    num_train = int(0.9 * len(score))

    scores, representations = shuffle(scores, representations)

    # Split data
    representations_train = representations[:num_train, :]
    scores_train = scores[:num_train]
    representations_test = representations[num_train:, :]
    scores_test = scores[num_train:]

    # Fit model
    reg = linear_model.LinearRegression()
    reg.fit(representations_train, scores_train)

    # Evaluate
    preds = reg.predict(representations_test)
    return pearsonr(preds, scores_test)

def linear_regression_cv(representations, scores, return_coef=False):
    scores, representations = shuffle(scores, representations)

    kf = KFold(n_splits=10)

    correlations = []
    p_values = []

    coefs = np.zeros((10, representations.shape[-1]))

    for i, (train_index, test_index) in enumerate(kf.split(representations)):
        # Split
        X_train, X_test = representations[train_index], representations[test_index]
        y_train, y_test = scores[train_index], scores[test_index]
        
        # Fit model
        reg = linear_model.LinearRegression()
        reg.fit(X_train, y_train)

        # Evaluate
        preds = reg.predict(X_test)
        res = pearsonr(preds, y_test)
        
        # Save fold correlation and p value
        correlations += [res[0]]
        p_values += [res[1]]
        
        coefs[i, :] =  reg.coef_

    # Take average of correlation and p value over folds
    avg_correlation = sum(correlations) / len(correlations)
    avg_p_value = sum(p_values) / len(p_values)

    if return_coef:
       return avg_correlation, avg_p_value, linea        

    return avg_correlation, avg_p_value
