#!/usr/bin/env python3
"""Retrieve and save data from/to csv. Read/Write from/to MongoDB.

Usage:

    python3 words.py <URL>
"""

import sys
import pandas as pd
from datetime import datetime
from pymongo import MongoClient


def read_csv(path, sep=',', header='infer'):
    """Read data from a csv file.

    Args:
        path: The path to the file.
        sep: Delimiter to use.
        header: Row number(s) to use as the column names.

    Returns:
        A pandas dataframe containing rows from the file.
        Throws Value exception if path is not good.
    """
    try:
        data = pd.read_csv(path, sep=sep, header=header)
        return data
    except OSError as e:
        print("Could not read file because {}".format(str(e)), file=sys.stderr)
        raise


def write_data_frame_to_csv(dir_path, file_name, data):
    """Write output to csv file. File name is formed from datetime stamp and file_name argument.

        Args:
            dir_path: The path to the directory.
            file_name: File name.
            data: Data to save to csv.
    """
    try:
        file_name = "".join([datetime.now().strftime("%Y%m%d%H%M%S"), '_', file_name, '.csv'])
        file_path = "\\".join([dir_path, file_name])

        data.to_csv(file_path, sep=',', header=True, index=False)

    except OSError as e:
        print("Could not write file because {}".format(str(e)), file=sys.stderr)
        raise


def write_pandas_to_mongo(dataframe, db, collection, host='localhost', port=27017, username=None, password=None):
    """Write Pandas dataframe to MongoDb document.

        Args:
            dataframe: Pandas DataFrame to be stored in MongoDb.
            db: Name of Mongo database.
            collection: Name of collection in which frame will be inserted.
            host: Mongo server host.
            port: Port on which Mongo server is listening.
            username: Username for MongoDb authentication.
            password: Password for MongoDb authentication.
    """
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)
    cursor = db[collection]

    cursor.insert_many(dataframe.to_dict('records'))


def write_dict_to_mongo(data_dict, db, collection, host='localhost', port=27017, username=None, password=None):
    """Write python dictionary to MongoDb document.

        Args:
            data_dict: Dictionary to be stored in MongoDb.
            db: Name of Mongo database.
            collection: Name of collection in which dictionary will be inserted.
            host: Mongo server host.
            port: Port on which Mongo server is listening.
            username: Username for MongoDb authentication.
            password: Password for MongoDb authentication.
    """
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)
    cursor = db[collection]

    cursor.insert_one(data_dict)


def delete_dict_from_mongo(db, collection, query={}, host='localhost', port=27017, username=None, password=None):
    """Delete documents from MongoDb collection which satisfy passed query.

        Args:
            db: Name of Mongo database.
            collection: Name of collection from which documents will be deleted.
            query: Criteria on which documents will be deleted.
            host: MongoDb server host.
            port: Port on which MongoDb server is listening.
            username: Username for MongoDb authentication.
            password: Password for MongoDb authentication.
    """
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)
    cursor = db[collection]

    cursor.delete_many(filter=query)


def update_mongo_collection(data_dict, db, collection, query={}, host='localhost', port=27017, username=None, password=None):
    """Update first document in MongoDb collection which satisfies the query.

        Args:
            db: Name of MongoDb database.
            collection: Name of collection where document will be updated.
            query: Criteria on which document from collection will be selected.
            host: MongoDb server host.
            port: Port on which MongoDb server is listening.
            username: Username for MongoDb authentication.
            password: Password for MongoDb authentication.
    """
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)
    db[collection].update_one(query, {"$set": data_dict}, upsert=False)


def read_pandas_from_mongo(db, collection, query={}, host='localhost', port=27017, username=None, password=None, no_id=True):
    """ Read documents from MongoDb collection which satisfy the query and retrieve it as Pandas DataFrame.
     
        Args:
            db: Name of MongoDb database.
            collection: Name of collection to be read.
            query: Criteria on which documents from collection will be selected.
            host: MongoDb server host.
            port: Port on which MongoDb server is listening.
            username: Username for MongoDb authentication.
            password: Password for MongoDb authentication.
            no_id: Flag to remove MongoDb document id.
            
        Returns:
            df: Pandas DataFrame with stored data.
    """
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    cursor = db[collection].find(query)

    df = pd.DataFrame(list(cursor))

    if no_id:
        del df['_id']

    return df


def read_dict_from_mongo(db, collection, query={}, host='localhost', port=27017, username=None, password=None, no_id=True):
    """ Read first document from MongoDb which satisfies the query and retrieve it as dict.
     
        Args:
            db: Name of MongoDb database.
            collection: Name of collection to be read.
            query: Criteria on which document from collection will be selected.
            host: MongoDb server host.
            port: Port on which MongoDb server is listening.
            username: Username for MongoDb authentication.
            password: Password for MongoDb authentication.
            no_id: Flag to remove MongoDb document id.
            
        Returns:
            data_dict: Dict with stored document.
    """
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    data_dict = db[collection].find_one(query)

    if data_dict != None and no_id:
        del data_dict['_id']

    if data_dict == None:
        return {}

    return data_dict


def _connect_mongo(host, port, username, password, db):
    """ Connect to MongoDb database and retrieve connection.

        Args:
            host: MongoDb server host.
            port: Port on which MongoDb server is listening.
            username: Username for MongoDb authentication.
            password: Password for MongoDb authentication.
            db: Name of MongoDb database.
            
        Returns:
            conn[db]: Connection to specified database.
    """
    if username and password:
        mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)

    return conn[db]


def mongo_collection_exists(db, collection, query={}, host='localhost', port=27017, username=None, password=None):
    """Check if MongoDb collection exists. If query is specified check if there are any documents within given collection 
        that satisfy given query.

        Args:
            db: Name of MongoDb database.
            collection: Name of collection to be read.
            query: Criteria on which document from collection will be selected.
            host: MongoDb server host.
            port: Port on which MongoDb server is listening.
            username: Username for MongoDb authentication.
            password: Password for MongoDb authentication.

        Returns:
            bool: True if collection/document exists, False otherwise.
    """
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    if len(query.keys()) == 0:
        return collection in db.collection_names()
    else:
        cursor = db[collection].find(query)
        return cursor.count() > 0


def get_collection_size(db, collection, query={}, host='localhost', port=27017, username=None, password=None):
    """Retrieve number of documents in a given MongoDb collection that satisfy the given criteria.

         Args:
            db: Name of MongoDb database.
            collection: Name of collection to be read.
            query: Criteria on which documents from collection will be selected.
            host: MongoDb server host.
            port: Port on which MongoDb server is listening.
            username: Username for MongoDb authentication.
            password: Password for MongoDb authentication.

        Returns:
            cursor.count(): Number of documents in collection. If collection does not exist this number is 0.
    """
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    if collection in db.collection_names():
        cursor = db[collection].find(query)
        return cursor.count()

    return 0
