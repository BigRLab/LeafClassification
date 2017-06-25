#!/usr/bin/env python3
"""Includes execution of classification algorithms from skicit learn package.

Usage:

    python3 words.py <URL>
"""

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def execute_classification(clf, data, target, split_ratio):
    """Split training set with skicit learn train_test_split function.
        Train classifier on training set and evaluate it on test set.

        Args:
            clf: Used Classifier.
            data: Available data attributes.
            target: Class attribute values.
            split_ratio: split ratio of data to be used for traning/testing.

        Returns:
            clf.score(testX, testY): Accuracy of evaluation on test data.

    """
    train_x, test_x, train_y, test_y = train_test_split(data, target, train_size=split_ratio)
    clf.fit(train_x, train_y)
    return clf.score(test_x, test_y)


def run_algorithm(clf, data, target, n_samples):
    """Execute classification 'n_samples' times. Each time available data set will
        be split differently to training and testing set.

        Args:
            clf: Used Classifier.
            data: Available data attributes.
            target: Class attribute values.
            n_samples: Number of times that classification will be executed.

        Returns:
            np.mean(score_array): Average accuracy of evaluation on test data in 'n_samples' iterations.
            np.std(score_array): Standard deviation of classification scores.

    """
    score_array = np.zeros(n_samples)

    for i in range(n_samples):
        score_array[i] = execute_classification(clf, data, target, 0.8)

    return np.mean(score_array), np.std(score_array)


def run_algorithm_with_pca(clf, data, target, n_pca_components_array, n_samples):
    """Execute classification 'n_samples' times as a function of each component in n_pca_components_array.
        Each time available data set will be split differently to training and testing set.

        Args:
            clf: Used Classifier.
            data: Available data attributes.
            target: Class attribute values.
            n_pca_components_array: Array of PCA components.
            n_samples: Number of times that classification will be executed.

        Returns:
            score_mean_array: Array of evaluation scores for each PCA component in 'n_samples' iterations.
            score_std_array: Array of standard deviation values of classification scores.

    """
    score_mean_array = np.zeros(len(n_pca_components_array))
    score_std_array = np.zeros(len(n_pca_components_array))

    for i in range(len(n_pca_components_array)):
        pca = PCA(n_components=n_pca_components_array[i])
        pca.fit(data)
        transformed_data = pca.transform(data)
        scores_iteration_array = np.zeros(n_samples)

        for j in range(n_samples):
            scores_iteration_array[j] = execute_classification(clf, transformed_data, target, 0.8)

        score_mean_array[i], score_std_array[i] = np.mean(scores_iteration_array), np.std(scores_iteration_array)

    return score_mean_array, score_std_array


def calculate_data_variance_ratio(n_pca_components_array, data):
    """Calculates the variance ratio that PCA algorithm captures with different number of PCA components.

        Args:
            n_pca_components_array: Array of PCA components.
            data: Available data set.

        Returns:
            vr: Array of variance ratio values.

    """
    vr = np.zeros(len(n_pca_components_array))
    i = 0

    for n_pca_component in n_pca_components_array:
        pca = PCA(n_components=n_pca_component)
        pca.fit(data)
        vr[i] = sum(pca.explained_variance_ratio_)
        i += 1

    return vr

