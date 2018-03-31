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
from scipy import sparse as sp

def load_data(num_negatives):
    """
    Loads and preprocessed data for the movielense 100k dataset.
    Returns input features, labels, and feature counts
    """
    # Load and preprocess data
    df_train, users_train, items_train = load_raw_data("../data/u1.base")
    df_valid, users_valid, items_valid = load_raw_data("../data/u1.test")

    # Generate feature and label lists
    user_train, item_train, label_train = get_train_instances(df_train, num_negatives)
    user_valid, item_valid, label_valid = get_train_instances(df_valid, num_negatives)

    return [[user_train, item_train, label_train], [user_valid, item_valid, label_valid]]

def load_raw_data(data_path):
    """
    Loads user/item interaction data from files
    Returns dataframe and user/feature counts
    """
    # Load data from files
    df = pd.read_csv(data_path, sep='\t', names=["user id", "item id", "rating", "timestamp"])
    df["implicit_feedback"] = 1

    # Compute number of unique users/items
    num_users = df[""].unique()
    num_items = df[""].unique()

    return df, num_users, num_items

def get_train_instances(train, num_negatives):
    """
    :param train: pandas df of training data
    :param num_negatives: number of negative examples to generate
    :return: 3 lists (user_list, item_list and label_list)
    """
    #convert to scipy sparse matrix of  user/item interaction data


    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in xrange(num_negatives):
            j = np.random.randint(num_items)
            while train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

def get_valid_instances(train, num_negatives):
    """
    clarify wht this should be???
    """
    pass