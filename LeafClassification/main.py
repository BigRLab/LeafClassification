"""Main module for testing DecisionTreeClassifier, KNeighborsClassifier,
    RandomForestClassifier and GaussianNB on LeafClassification problem from kaggle.

Usage:

    python3 words.py <URL>

"""

import sys

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from Constants.constants import *
from Common.common import *
from Algorithms.algorithms import *
from Data.data import *


def execute_algorithms(train_data, target_data, n_samples, data_scrubbing_description):
    """Executes Decision Tree, Random Forest, KNeighbors and GaussianNB algorithms on
        raw or standardized training data.
        Prints results of classification.
        Plots results of Random Forest algorithm as a function of number of estimators.

        Args:
            train_data: Available data attributes with values.
            target_data: Class attribute values.
            n_samples: Number of times that classification will be executed.
            data_scrubbing_description: Description on what data is being used.

    """
    dt_clf = DecisionTreeClassifier()
    kn_clf = KNeighborsClassifier()
    nb_clf = GaussianNB()
    n_estimators_array = np.array([1, 5, 10, 50, 100, 200])

    dt_score, dt_score_std = run_algorithm(dt_clf, train_data, target_data, n_samples)
    print_single_section("Decision Tree Classifier", data_scrubbing_description, dt_score)

    rf_score_array = np.zeros(len(n_estimators_array))
    rf_score_std_array = np.zeros(len(n_estimators_array))
    for i in range(len(n_estimators_array)):
        rf_clf = RandomForestClassifier(n_estimators = n_estimators_array[i])
        rf_score_array[i], rf_score_std_array[i] = run_algorithm(rf_clf, train_data, target_data, n_samples)

    print_multiple_section("Random Forest Classifier", data_scrubbing_description,
                           "For n_estimators = {0:0=3d} mean accuracy is {1:.6f}", n_estimators_array, rf_score_array)
    plot_chart("Number of estimators", "accuracy", n_estimators_array, rf_score_array,
               title_text='Random Forest Classifier ' + data_scrubbing_description, sigma_component=rf_score_std_array)

    kn_score, kn_std = run_algorithm(kn_clf, train_data, target_data, 5)
    print_single_section("KNeighbours Classifier", data_scrubbing_description, kn_score)

    nb_score, nb_std = run_algorithm(nb_clf, train_data, target_data, 5)
    print_single_section("Naive Bayes Classifier", data_scrubbing_description, nb_score)


def execute_pca_variance_calculation(train_data):
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
    n_components_array = ([1, 5, 10, 20, 50, 100, 150, 180])

    vr = calculate_data_variance_ratio(n_components_array, train_data)
    plot_pca_chart("Number of PCA components", "variance ratio", n_components_array, vr)


def execute_algorithms_with_pca(train_data, target_data, n_samples, data_scrubbing_description):
    """Executes Decision Tree, Random Forest, KNeighbors and GaussianNB algorithms on
        raw or standardized training data as a function of number of PCA components.
        Prints results of classification.
        Plots results of all algorithms as a function of number of PCA components.

        Args:
            train_data: Available data attributes with values.
            target_data: Class attribute values.
            n_samples: Number of times that classification will be executed.
            data_scrubbing_description: Description on what data is being used.

    """
    dt_clf = DecisionTreeClassifier()
    rf_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    kn_clf = KNeighborsClassifier()
    nb_clf = GaussianNB()
    n_components_array = ([1, 5, 10, 20, 50, 100, 150, 180])

    dt_score_array, dt_score_std_array = run_algorithm_with_pca(dt_clf, train_data, target_data, n_components_array, n_samples)
    print_multiple_section("Decision Tree Classifier + PCA Decomposition", data_scrubbing_description,
                           "For {0:0=3d} PCA components mean accuracy is {1:.6f}", n_components_array, dt_score_array)
    plot_chart('number of PCA components', 'accuracy', n_components_array, dt_score_array,
               title_text='Decision Tree Classifier ' + data_scrubbing_description, sigma_component=dt_score_std_array)

    rf_score_array, rf_score_std_array = run_algorithm_with_pca(rf_clf, train_data, target_data, n_components_array, n_samples)
    print_multiple_section("Random Forest Classifier + PCA Decomposition", data_scrubbing_description,
                           "For n_estimators = {0:0=3d} mean accuracy is {1:.6f}", n_components_array, rf_score_array)
    plot_chart('number of PCA components', 'accuracy', n_components_array, rf_score_array,
               title_text='Random Forest Classifier ' + data_scrubbing_description, sigma_component=rf_score_std_array)

    kn_score_array, kn_score_std_array = run_algorithm_with_pca(kn_clf, train_data, target_data, n_components_array, n_samples)
    print_multiple_section("KNeigbors Classifier + PCA Decomposition", data_scrubbing_description,
                           "For {0:0=3d} PCA components mean accuracy is {1:.6f}", n_components_array, kn_score_array)
    plot_chart('number of PCA components', 'accuracy', n_components_array, kn_score_array,
               title_text='KNeigbors Classifier ' + data_scrubbing_description, sigma_component=kn_score_std_array)

    nb_score_array, nb_score_std_array = run_algorithm_with_pca(nb_clf, train_data, target_data, n_components_array, n_samples)
    print_multiple_section("Naive Bayes Classifier + PCA Decomposition", data_scrubbing_description,
                           "For {0:0=3d} PCA components mean accuracy is {1:.6f}", n_components_array, nb_score_array)
    plot_chart('number of PCA components', 'accuracy', n_components_array, nb_score_array,
               title_text='Naive Bayes Classifier ' + data_scrubbing_description, sigma_component=nb_score_std_array)

    plot_multiple_charts(x_axis_array=n_components_array,
                          errorbar_two_dim_mean_array=[dt_score_array,
                                                       rf_score_array,
                                                       kn_score_array,
                                                       nb_score_array],
                          errorbar_two_dim_std_array=[dt_score_std_array,
                                                       rf_score_std_array,
                                                       kn_score_std_array,
                                                       nb_score_std_array],
                          x_label="num PCA components",
                          y_label="validation accuracy",
                          title_text="Algorithms " + data_scrubbing_description,
                          legend=['Decision Tree', 'Random Forest', 'k Nearest Neighbor', 'Naive Bayes'])


def main(n_samples):
    """Main function.
        Loads train and test data and invokes execution of classification algorithms.

        Args:
            n_samples: Number of times that classification will be executed.

    """
    train = read_csv(TRAIN_FILE_PATH)
    #test = read_csv(TEST_FILE_PATH)
    target = train['species']
    train = train.drop(['id', 'species'], 1)
    scaler = StandardScaler().fit(train)
    train_standardized = scaler.transform(train)

    print_header("Execute classification without data processing")
    execute_algorithms(train, target, n_samples, "without any data preprocessing")

    print_header("Execute classification with data standardization")
    execute_algorithms(train_standardized, target, n_samples, "with data standardization")

    print_header("Capture training data variance with PCA")
    execute_pca_variance_calculation(train)

    print_header("Execute classification after data decomposition")
    execute_algorithms_with_pca(train, target, n_samples, "with data decomposition")

    print_header("Execute classification after data decomposition and standardization")
    execute_algorithms_with_pca(train_standardized, target, n_samples, "with data decomposition and standardization")


if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except IndexError as e:
        print("Use default number of samples = 2")
        main(2)
