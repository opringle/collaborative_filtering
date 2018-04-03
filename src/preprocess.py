#!/usr/bin/env python

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

# -*- coding: utf-8 -*-

import numpy as np
import argparse
import logging
from Dataset import Dataset
import os

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Script to preprocess data",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', nargs='?', default='../Data/',
                        help='Input data path.')
parser.add_argument('--output_path', nargs='?', default='../preprocessed_data/',
                        help='Output preprocessed data path.')
parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
parser.add_argument('--num-negatives', type=int, default=4,
                        help='Number of negative examples per user in training set.')

def get_train_instances(train, num_negatives):
    """
    Read in scipy  sparse matrix
    :return: 3 arrays of input data
    """

    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

def get_test_instances(test_ratings, test_negatives):
    """
    Read in scipy  sparse matrix
    :return: 3 arrays of input data
    """
    user_input, item_input, labels = [], [], []
    for i, (user, item) in enumerate(test_ratings):
        user_input.append(user)
        item_input.append(item)
        labels.append(1)
        #negatives
        user_input.extend([user]*len(test_negatives[i]))
        item_input.extend(test_negatives[i])
        labels.extend([0] * len(test_negatives[i]))
    return user_input, item_input, labels



if __name__ == '__main__':
    # parse args
    args = parser.parse_args()

    # create a dataset object and retrieve train test data
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape

    # save user, item numbers to file
    f = open(os.path.join(args.output_path, "users_items.txt"), "w")
    f.write(str((num_users, num_items)))
    f.close()

    X_train_user, X_train_item, Y_train = get_train_instances(train, num_negatives=args.num_negatives)
    X_test_user, X_test_item, Y_test = get_test_instances(testRatings, testNegatives)

    # save numpy files to array
    np.save(os.path.join(args.output_path, "X_train_user"), X_train_user)
    np.save(os.path.join(args.output_path, "X_train_item"), X_train_item)
    np.save(os.path.join(args.output_path, "Y_train"), Y_train)
    np.save(os.path.join(args.output_path, "X_test_user"), X_test_user)
    np.save(os.path.join(args.output_path, "X_test_item"), X_test_item)
    np.save(os.path.join(args.output_path, "Y_test"), Y_test)
