"""Main module for testing DecisionTreeClassifier, KNeighborsClassifier,
    RandomForestClassifier and GaussianNB on LeafClassification problem from kaggle.

Usage:

    python3 words.py <URL>

"""

import Algorithms.business as bl
import Common.constants as const
import Data.data as dl
import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder


def initialize_mongo_db(version=1):
    """Initializes MongoDb, reads train.csv and test.csv and stores them into LC_TRAIN_DATA and
        LC_TEST_DATA collections in LC database. Initializes LC_SYSTEM collection with version 1 and
        with number of records in LC_TRAIN_DATA collection.
    """
    if not bl.mongo_collection_exists("LC", "LC_TRAIN_DATA"):
        train_data = dl.read_csv(const.TRAIN_FILE_PATH)
        print("--Read train.csv file.")
        dl.write_pandas_to_mongo(train_data, "LC", "LC_TRAIN_DATA")
        print("--Store contents from train.csv to MongoDB.")

    if not bl.mongo_collection_exists("LC", "LC_TEST_DATA"):
        test_data = dl.read_csv(const.TEST_FILE_PATH)
        print("--Read test.csv file.")
        dl.write_pandas_to_mongo(test_data, "LC", "LC_TEST_DATA")
        print("--Store contents from test.csv to MongoDB.")

    if not bl.mongo_collection_exists("LC", "LC_SYSTEM", {"version": version}):
        dl.write_dict_to_mongo({"version": version, "train_collection_count": dl.get_collection_size(db="LC", collection="LC_TRAIN_DATA")}, "LC", "LC_SYSTEM")
        print("--Initialize system collection in MongoDB, version {0}.".format(version))

    print("\n\n")

def main(version):
    """Main function.
        Loads train and test data and invokes execution of classification algorithms.
        Initializes MongoDb.
    """
    initialize_mongo_db(version=version)

    if bl.persisted_models_not_valid(system_query={"version": version}):
        print("--Train data set changed, models for version {0} need to be retrained".format(version))
        bl.reset_system_collection(system_query={"version": version})
        print("\n")

    print("--Read test data from MongoDB.")
    test_data = dl.read_pandas_from_mongo("LC", "LC_TEST_DATA")
    test_data_ids = [int(i) for i in test_data["id"]]
    test_data = test_data.drop(["id"], 1)

    le = LabelEncoder().fit(dl.read_pandas_from_mongo("LC", "LC_TRAIN_DATA")['species'])

    #load scaler and standardize test data
    scaler = bl.get_standard_scaler(version=version)
    standardized_test_data = scaler.transform(test_data)

    print("\n")
    #Random Forest
    print("--Evaluate test data with Random Forest.")
    pca = bl.get_pca_scaler(n_components=50, version=version)
    transformed_test_data = pca.transform(standardized_test_data)
    clf = bl.get_randomforest_classifier(n_trees=500, version=version)
    predictions = clf.predict_proba(transformed_test_data)

    submission = pd.DataFrame(predictions, columns=le.classes_)
    submission.insert(0, "id", test_data_ids)
    print("--Write predictions to random_forest_v{0}.csv\n\n".format(version))
    dl.write_data_frame_to_csv(dir_path="\\".join([const.DIR_PATH, "Predictions"]), file_name="random_forest_v{0}".format(version), data=submission)

    #Decision Tree
    print("--Evaluate test data with Decision Tree.")
    pca = bl.get_pca_scaler(n_components=10, version=version)
    transformed_test_data = pca.transform(standardized_test_data)
    clf = bl.get_decisiontree_classifier(version=version)
    predictions = clf.predict_proba(transformed_test_data)

    submission = pd.DataFrame(predictions, columns=le.classes_)
    submission.insert(0, "id", test_data_ids)
    print("--Write predictions to decision_tree_v{0}.csv\n\n".format(version))
    dl.write_data_frame_to_csv(dir_path="\\".join([const.DIR_PATH, "Predictions"]), file_name="decision_treet_v{0}".format(version), data=submission)

    #Naive Bayes
    print("--Evaluate test data with Naive Bayes.")
    pca = bl.get_pca_scaler(n_components=20, version=version)
    transformed_test_data = pca.transform(standardized_test_data)
    clf = bl.get_naivebayes_classifier(version=version)
    predictions = clf.predict_proba(transformed_test_data)
    
    submission = pd.DataFrame(predictions, columns=le.classes_)
    submission.insert(0, "id", test_data_ids)
    print("--Write predictions to naive_bayes_v{0}.csv\n\n".format(version))
    dl.write_data_frame_to_csv(dir_path="\\".join([const.DIR_PATH, "Predictions"]), file_name="naive_bayes_v{0}".format(version), data=submission)
    
    #KNeighbors
    print("--Evaluate test data with KNeighbors.")
    pca = bl.get_pca_scaler(n_components=50, version=version)
    transformed_test_data = pca.transform(standardized_test_data)
    clf = bl.get_kneighbors_classifier(version=version)
    predictions = clf.predict_proba(transformed_test_data)
    
    submission = pd.DataFrame(predictions, columns=le.classes_)
    submission.insert(0, "id", test_data_ids)
    print("--Write predictions to k_neighbors_v{0}.csv\n\n".format(version))
    dl.write_data_frame_to_csv(dir_path="\\".join([const.DIR_PATH, "Predictions"]), file_name="k_neighbors_v{0}".format(version), data=submission)


if __name__ == '__main__':
    try:
        main(version=sys.argv[1])
    except IndexError as e:
        defaultVersion = 1
        print("Use default version: {0}\n\n".format(defaultVersion))
        main(version=defaultVersion)
