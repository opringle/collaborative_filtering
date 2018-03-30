# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import pandas as pd
import numpy as np

def load_data():
    """
    Loads and preprocessed data for the movielense 100k dataset.
    Returns input features, labels, and feature counts
    """
    # Load and preprocess data
    x, y = load_raw_data()

    # Create dictionaries for categorical input features
    feature_counts = feature_count(x)
    return [x, y, feature_counts]

def load_raw_data():
    """
    Loads user/item interaction data from files, splits the data into features and labels.
    Returns a list of numpy arrays containing features and labels
    """
    # Load data from files
    data_path = "../data/u.data"
    df = pd.read_csv(data_path, sep='\t', names=["user id", "item id", "rating", "timestamp"])

    #convert to numpy arrays
    x = df.as_matrix(columns=["user id", "item id"])
    y = df.as_matrix(columns=["rating"])

    return [x, y]

def feature_count(feature_array):
    """
    Counts number of unique features in each column of a 2d input array.
    Returns a dictionary of feature counts per column.
    """
    feature_sizes = {}
    for i in range(feature_array.shape[1]):
        num_unique_elements = len(np.unique(feature_array[:,i]))
        feature_sizes["feature_" + str(i)] = num_unique_elements

    return feature_sizes