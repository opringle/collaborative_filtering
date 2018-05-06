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

import mxnet as mx
import numpy as np
import scipy.sparse as sps
from operator import itemgetter
from sklearn import utils
import time

# ToDo: Negative sampling takes 10s per epoch.


class SparseNegativeSamplingDataIter(mx.io.DataIter):
    """DataIter for negative sampling from a sparse matrix"""
    def __init__(self, sparse_interactions, user_features, item_features, negatives_per_interaction, interaction_label,
                 interaction_labels, negative_sample_label, batch_size, data_names=["user_x", "item_x"],
                 label_names=["softmax_label"], create_batches=True, shuffle=True, resample=False):
        """

        :param sparse_interactions:
        :param user_features:
        :param item_features:
        :param negatives_per_interaction:
        :param interaction_label:
        :param interaction_labels:
        :param negative_sample_label:
        :param batch_size:
        :param data_names:
        :param label_names:
        :param create_batches:
        :param shuffle:
        """
        self.resample=resample
        self.shuffle=shuffle
        self.sparse_interactions = sparse_interactions
        self.create_batches = create_batches
        self.data_names = data_names
        self.label_names = label_names
        self.batch_size = batch_size
        self.negatives = negatives_per_interaction
        self.interaction_label = interaction_label
        self.interaction_labels = interaction_labels
        self.negative_sample_label = negative_sample_label

        # Create a list of mx.ndarray of user features & item features to sample when producing a batch
        self.unique_user_features = mx.nd.array(user_features)
        self.unique_item_features = mx.nd.array(item_features)

        self.provide_data = [mx.io.DataDesc(name=self.data_names[0],
                                            shape=(self.batch_size, self.unique_user_features.shape[1])),
                             mx.io.DataDesc(name=self.data_names[1],
                                            shape=(self.batch_size, self.unique_item_features.shape[1]))]
        self.provide_label = [mx.io.DataDesc(name=self.label_names[0], shape=(self.batch_size, ))]

        self.generate_negative_samples() # when initializing generate negative samples
        self.reset()

    def generate_negative_samples(self):
        start =  time.time()
        # Create a list (row, col) to sample positive indices from
        self.users, self.items, self.labels = [], [], []

        # Generate random samples
        for i, user in enumerate(self.sparse_interactions):
            # Get item interactions for that user
            other_interactions = []
            for label in self.interaction_labels:  # if we find an interaction check if it matches the interaction label
                if label == self.interaction_label:
                    interactions = user.multiply(user == label).tolil().rows[0]
                else:
                    other_interactions.extend(user.multiply(user == label).tolil().rows[0])

            self.users.extend([i] * len(interactions))
            self.items.extend(interactions)
            self.labels.extend([1] * len(interactions))

            # Get x negatives per interaction for that user
            user_negatives = set(list(range(0, self.sparse_interactions.shape[1]))) - set(interactions) - \
                             set(other_interactions)
            sample_size = int(min(self.negatives * len(interactions), len(user_negatives)))
            selected_negatives = np.random.choice(list(user_negatives), size=sample_size, replace=False)

            self.users.extend([i] * len(selected_negatives))
            self.items.extend(selected_negatives)
            self.labels.extend([0] * len(selected_negatives))

        # Set sparse array values where we sampled negatives (so we have a record of what data is left unsampled)
        negative_indices = [i for i, l in enumerate(self.labels) if l == 0]
        negative_users = np.array(itemgetter(*negative_indices)(self.users))
        negative_items = np.array(itemgetter(*negative_indices)(self.items))
        data = np.ones(shape=(len(negative_indices),)) * self.negative_sample_label

        sparse_sampled_negatives = sps.csr_matrix((data, (negative_users, negative_items)),
                                                  shape=self.sparse_interactions.shape)
        self.sparse_interactions = self.sparse_interactions + sparse_sampled_negatives

        print("negative sample time: {}".format(time.time()-start))

    def reset(self):
        """Resets the iterator to the beginning of the data. Generates new random negative samples."""
        start = time.time()
        self.curr_idx = 0

        if self.resample:
            self.generate_negative_samples()

        # Shuffle data
        if self.shuffle:
            self.users, self.items, self.labels = utils.shuffle(self.users, self.items, self.labels)

        # Organize coords into batches
        self.idx = []
        self.idx.extend([j for j in range(0, len(self.labels) - self.batch_size + 1, self.batch_size)])
        self.curr_idx = 0

        self.ndlabels = mx.nd.array(self.labels)
        self.ndusers = mx.nd.array(self.users)
        self.nditems = mx.nd.array(self.items)

        # Create feature data
        if self.create_batches:
            self.nduserfeatures = mx.ndarray.take(a=self.unique_user_features, indices=self.ndusers)
            self.nditemfeatures = mx.ndarray.take(a=self.unique_item_features, indices=self.nditems)

        print("iterator reset time: {}".format(time.time() - start))

    def next(self):
        """Returns the next batch of data."""
        if self.curr_idx == len(self.idx):
            raise StopIteration

        # Fetch the index
        i = self.idx[self.curr_idx]
        self.curr_idx += 1

        # Get labels & ids
        labels = self.ndlabels[i:i+self.batch_size]
        users = self.ndusers[i:i + self.batch_size]
        items = self.nditems[i:i + self.batch_size]

        # Get feature arrays
        if self.create_batches:
            user_features = self.nduserfeatures[i:i+self.batch_size]
            item_features = self.nditemfeatures[i:i+self.batch_size]
        else:
            # Create user feature arrays
            user_features = mx.ndarray.take(a=self.unique_user_features, indices=users)
            item_features = mx.ndarray.take(a=self.unique_item_features, indices=items)

        return mx.io.DataBatch([user_features, item_features], [labels], index = users, pad=0,
                         provide_data=[mx.io.DataDesc(name=self.data_names[0], shape=user_features.shape),
                                       mx.io.DataDesc(name=self.data_names[1], shape=item_features.shape)],
                         provide_label=[mx.io.DataDesc(name=self.label_names[0], shape=labels.shape)])