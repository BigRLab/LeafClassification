#!/usr/bin/env python3
"""Retrieve and save data from/to csv.

Usage:

    python3 words.py <URL>
"""

import sys
import pandas as pd
import numpy as np
import datetime


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


def write_csv(dir_path, file_name, data, columns):
    """Write output to csv file. File name is formed from datetime stamp and file_name argument.

        Args:
            dir_path: The path to the directory.
            file_name: File name.
            data: Data to save to csv.
            columns: Headers for data to be saved.
    """
    try:
        file_name = "".join([str(datetime), '_', file_name, '.csv'])
        file_path = "\\".join([dir_path, file_name])

        np.savetxt(
            file_path,
            np.c_[range(1, len(data) + 1), data],
            delimiter=',',
            header=','.join([columns]),
            comments='',
            fmt='%d')
    except OSError as e:
        print("Could not read file because {}".format(str(e)), file=sys.stderr)
        raise
