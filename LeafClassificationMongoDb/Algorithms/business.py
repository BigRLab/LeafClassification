#!/usr/bin/env python3
"""Business Logic for training and preserving classification algorithms from skicit learn package.

Usage:

    python3 words.py <URL>
"""

import pickle
import Data.data as dl
from Common import constants
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def get_standard_scaler(version=1):
    """Retrieves StandardScaler from scikit package. Scaler is first trained and then preserved in MongoDb.
        Once preserved Scaler is retrieved from MongoDb on all following function calls.

        Returns:
            scaler: sklearn.preprocessing StandardScaler
    """
    system = dl.read_dict_from_mongo("LC", "LC_SYSTEM", {"version": version})

    if "scaler_preserved" in system and system["scaler_preserved"]:
        print("--StandardScaller already preserved.")

        return pickle.loads(system["scaler_model_bin"])
    else:
        print("--StandardScaller trained and preserved for the first time.")

        #load train data
        train_data, _ = get_lc_train_data()

        #train and preserve scaler
        scaler = StandardScaler().fit(train_data)

        system["scaler_model_bin"] = pickle.dumps(scaler)
        system["scaler_preserved"] = True
        dl.update_mongo_collection(system, "LC", "LC_SYSTEM", {"version": version})

        return scaler


def get_decisiontree_classifier(version=1):
    """Retrieves Decision Tree classifier from scikit package. Classifier is first trained and 
        then preserved on file system. Once preserved classifier is retrieved from file system on all following function calls.
        Classifier is trained on training data first standardized with StandardScaler and then transformed with PCA(n_components=10)
        
        Returns:
            dt_clf: sklearn.tree DecisionTreeClassifier
    """
    system = dl.read_dict_from_mongo("LC", "LC_SYSTEM", {"version": version})

    if "dt_preserved" in system and system["dt_preserved"]:
        print("--Decision Tree model already preserved.")

        with open(system["dt_model_disk_path"], "rb") as model_file:
            return pickle.load(model_file)
    else:
        print("--DecisionTree trained and preserved for the first time.")

        #load train data
        train_data, target_classes = get_lc_train_data()

        #load scaler and transform data
        scaler = get_standard_scaler(version=version)
        standardized_train_data = scaler.transform(train_data)

        #load PCA and transform data
        pca = get_pca_scaler(n_components=10, version=version)
        transformed_train_data = pca.transform(standardized_train_data)

        #train and preserve Decision Tree
        dt_clf = DecisionTreeClassifier()
        dt_clf.fit(transformed_train_data, target_classes)
        model_file_path = _get_custom_clf_model_file_path("dt_model_v{0}".format(version))
        with open(model_file_path, "wb") as model_file:
            pickle.dump(dt_clf, model_file)

        system["dt_preserved"] = True
        system["dt_model_disk_path"] = model_file_path
        dl.update_mongo_collection(system, "LC", "LC_SYSTEM", {"version": version})

        return dt_clf


def get_randomforest_classifier(n_trees=500, version=1):
    """Retrieves Random Forest classifier from scikit package. Classifier is first trained and 
        then preserved on file system. Once preserved classifier is retrieved from file system on all following function calls.
        Classifier is trained on training data first standardized with StandardScaler and then transformed with PCA(n_components=50)
        
        Args:
            n_trees: Number of trees for Random Forest classifier.
        
        Returns:
            rf_clf: sklearn.ensemble import RandomForestClassifier
    """

    system = dl.read_dict_from_mongo("LC", "LC_SYSTEM", {"version": version})

    if "rf{0}_preserved".format(n_trees) in system and system["rf{0}_preserved".format(n_trees)]:
        print("--RandomForest with {0} trees already preserved.".format(n_trees))

        with open(system["rf{0}_model_disk_path".format(n_trees)], "rb") as model_file:
            return pickle.load(model_file)
    else:
        print("--Random Forest with {0} trees trained and preserved for the first time.".format(n_trees))

        #load train data
        train_data, target_classes = get_lc_train_data()

        #load scaler and standardize data
        scaler = get_standard_scaler(version=version)
        standardized_train_data = scaler.transform(train_data)

        #load PCA and transform data
        pca = get_pca_scaler(n_components=50, version=version)
        transformed_train_data = pca.transform(standardized_train_data)

        #train and preserve RandomForest
        rf_clf = RandomForestClassifier(n_estimators=n_trees)
        rf_clf.fit(transformed_train_data, target_classes)
        model_file_path = _get_custom_clf_model_file_path("rf{0}_model_v{1}".format(n_trees, version))
        with open(model_file_path, "wb") as model_file:
            pickle.dump(rf_clf, model_file)

        system["rf{0}_preserved".format(n_trees)] = True
        system["rf{0}_model_disk_path".format(n_trees)] = model_file_path
        dl.update_mongo_collection(system, "LC", "LC_SYSTEM", {"version": version})

        return rf_clf


def get_kneighbors_classifier(version=1):
    """Retrieves KNeighbors classifier from scikit package. Classifier is first trained and 
        then preserved on file system. Once preserved classifier is retrieved from file system on all following function calls.
        Classifier is trained on training data first standardized with StandardScaler and then transformed with PCA(n_components=50)

        Args:
            n_trees: Number of trees for Random Forest classifier.

        Returns:
            kn_clf: sklearn.neighbors KNeighborsClassifier
    """
    system = dl.read_dict_from_mongo("LC", "LC_SYSTEM", {"version": version})

    if "kn_preserved" in system and system["kn_preserved"]:
        print("--KNeighbors already preserved.")

        with open(system['kn_model_disk_path'], 'rb') as model_file:
            return pickle.load(model_file)
    else:
        print("--KNeighbors trained and preserved for the first time.")

        #load train data
        train_data, target_classes = get_lc_train_data()

        #load scaler and transform data
        scaler = get_standard_scaler(version=version)
        standardized_train_data = scaler.transform(train_data)

        # load PCA and transform data
        pca = get_pca_scaler(n_components=50, version=version)
        transformed_train_data = pca.transform(standardized_train_data)

        #train and preserve KNeighbors
        kn_clf = KNeighborsClassifier()
        kn_clf.fit(transformed_train_data, target_classes)
        model_file_path = _get_custom_clf_model_file_path("kn_model_v{0}".format(version))
        with open(model_file_path, "wb") as model_file:
            pickle.dump(kn_clf, model_file)

        system["kn_preserved"] = True
        system["kn_model_disk_path"] = model_file_path
        dl.update_mongo_collection(system, "LC", "LC_SYSTEM", {"version": version})

        return kn_clf


def get_naivebayes_classifier(version=1):
    """Retrieves Naive Bayes classifier from scikit package. Classifier is first trained and 
        then preserved on file system. Once preserved classifier is retrieved from file system on all following function calls.
        Classifier is trained on training data first standardized with StandardScaler and then transformed with PCA(n_components=20)

        Returns:
            nb_clf: sklearn.naive_bayes GaussianNB
    """
    system = dl.read_dict_from_mongo("LC", "LC_SYSTEM", {"version": version})

    if "nb_preserved" in system and system['nb_preserved']:
        print("--Naive Bayes already preserved.")

        with open(system['nb_model_disk_path'], 'rb') as model_file:
            return pickle.load(model_file)
    else:
        print("--Naive Bayes trained and preserved for the first time.")

        #load train data
        train_data, target_classes = get_lc_train_data()

        #load scaler and transform data
        scaler = get_standard_scaler(version=version)
        standardized_train_data = scaler.transform(train_data)

        # load PCA and transform data
        pca = get_pca_scaler(n_components=20, version=version)
        transformed_train_data = pca.transform(standardized_train_data)

        #train and preserve Naive Bayes
        nb_clf = GaussianNB()
        nb_clf.fit(transformed_train_data, target_classes)
        model_file_path = _get_custom_clf_model_file_path("nb_model{0}".format(version))
        with open(model_file_path, "wb") as model_file:
            pickle.dump(nb_clf, model_file)

        system["nb_preserved"] = True
        system["nb_model_disk_path"] = model_file_path
        dl.update_mongo_collection(system, "LC", "LC_SYSTEM", {"version": version})

        return nb_clf


def get_pca_scaler(n_components, svd_solver="auto", version=1):
    """Retrieves PCA from scikit package. PCA is first trained and 
        then preserved on file system. Once preserved PCA is retrieved from file system on all following function calls.
        PCA is trained on training data first standardized with StandardScaler.
        
        Args:
            n_components: Number of components to retrieve with PCA.
            svd_solver: PCA svd_solver

        Returns:
            pca: sklearn.decomposition PCA
    """
    system = dl.read_dict_from_mongo("LC", "LC_SYSTEM", {"version": version})

    if str("pca{0}_preserved".format(n_components)) in system and system["pca{0}_preserved".format(n_components)]:
        print("--PCA (n_components={0}) already preserved.".format(n_components))

        with open(system["pca" + str(n_components) + "_model_disk_path"], "rb") as model_file:
            return pickle.load(model_file)
    else:
        print("--PCA (n_components={0}) trained and preserved for the first time.".format(n_components))

        # load train data
        train_data, _ = get_lc_train_data()

        # load scaler and transform data
        scaler = get_standard_scaler(version=version)
        standardized_train_data = scaler.transform(train_data)

        # train and preserve PCA
        pca = PCA(n_components=n_components, svd_solver=svd_solver)
        pca.fit(standardized_train_data)
        model_file_path = _get_custom_clf_model_file_path("pca{0}_model_v{1}".format(n_components, version))
        with open(model_file_path, "wb") as model_file:
            pickle.dump(pca, model_file)

        system["pca{0}_preserved".format(n_components)] = True
        system["pca{0}_model_disk_path".format(n_components)] = model_file_path
        dl.update_mongo_collection(system, "LC", "LC_SYSTEM", {"version": version})

        return pca


def get_lc_train_data():
    """Gets train data and target classes from MongoDb LC database.

        Returns:
            train_data: Pandas DataFrame with train data.
            target_classes: List of target classes for retrieved train records.
    """
    train_data = dl.read_pandas_from_mongo("LC", "LC_TRAIN_DATA")
    target_classes = train_data["species"]
    train_data = train_data.drop(["id", "species"], 1)

    return train_data, target_classes


def mongo_collection_exists(db, collection_name, query={}):
    """Check if MongoDb collection exists. If query is specified check if there are any documents within given collection 
        that satisfy given query.

        Returns:
            bool: True if collection/document exists, False otherwise.
    """
    return dl.mongo_collection_exists(db=db, collection=collection_name, query=query)


def persisted_models_not_valid(system_query={}):
    """Checks if state of persisted models is valid. If train data set is changed since models were trained
        their state is not valid and they need to be trained again to reflect latest dataset.

        Returns:
            bool: True if models are not valid, False if models are valid.
    """
    train_collection_count = dl.get_collection_size(db="LC", collection="LC_TRAIN_DATA")
    system = dl.read_dict_from_mongo(db="LC", collection="LC_SYSTEM", query=system_query)

    if "train_collection_count" in system:
        return int(system["train_collection_count"]) != int(train_collection_count)

    return False


def reset_system_collection(system_query, db="LC", collection="LC_SYSTEM"):
    """Deletes the first system collection that satisfies the system_query and then creates new one
        with the same system_query and number of records in train set.
    """
    print("--Removing document from LC_SYSTEM collection.")
    dl.delete_dict_from_mongo(db=db, collection=collection, query=system_query)

    document = system_query
    document["train_collection_count"] = dl.get_collection_size(db=db, collection="LC_TRAIN_DATA")

    print("--Adding updated document to LC_SYSTEM collection.")
    dl.write_dict_to_mongo(db=db, collection=collection, data_dict=document)


def _get_custom_clf_model_file_path(filename):
    """Gets custom file path by adding %Y%m%d%H%M%S prefix to the filename.
    
        Args:
            filename
        
        Returns:
            str: FileName in format %Y%m%d%H%M%S_filename.pkl
    """
    file_name = "".join([datetime.now().strftime("%Y%m%d%H%M%S"), "_", filename, ".pkl"])
    return "".join([constants.TRAINED_MODELS_PATH, file_name])
