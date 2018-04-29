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

import pandas as pd
import scipy.sparse as sps
import mxnet as mx
import numpy as np
import argparse
import logging
import time
import os
from libs import iterators
from libs import metrics

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Neural Collaborative Filtering Model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', nargs='?', default='./data',
                        help='Input data folder')
parser.add_argument('--test-interactions', type=int, default=1,
                    help='for each user latest n interactions are test set')
parser.add_argument('--train-negatives', type=int, default=2,
                    help='the number of negative samples per training interaction')
parser.add_argument('--test-negatives', type=int, default=99,
                    help='the number of negative samples per validation interaction')
parser.add_argument('--topk', type=int, default=10,
                    help='size of model ranking list')
parser.add_argument('--batch-size', type=int, default=256,
                    help='the number of training records in each minibatch')
parser.add_argument('--num-epochs', type=int, default=100,
                    help='how  many times to update the model parameters')
parser.add_argument('--num-embed', type=int, default=64,
                    help='the user/item latent feature dimension')
parser.add_argument('--fc-layers', type=list, default=[64,32,16],
                    help='list of hidden layer sizes')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='the user/item latent feature dimension')
parser.add_argument('--optimizer', type=str, default='Adam',
                    help='optimization algorithm to update model parameters with')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate for chosen optimizer')
parser.add_argument('--output-dir', type=str, default='checkpoint',
                    help='directory to save model params/symbol to')
parser.add_argument('--clean-output-dir', type=bool, default=True,
                    help='delete previous output directory files')
parser.add_argument('--gpus', type=str, default='',
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ')

def evaluate(module, iterator, k, test_interactions, num_users):
    """
    Evaluate a module and iterator
    """
    def _callback(epoch, sym=None, arg=None, aux=None):
        if epoch%10==0 and epoch>0:
            return metrics.TopKAccuracy(module, iterator, k, test_interactions, num_users)

    return _callback

def reset_id(df, cols):

    for col in cols:
        old_ids=df[col].unique().tolist()
        new_ids=list(range(len(old_ids)))
        mapping = dict(zip(old_ids, new_ids))
        df.loc[:, col] = [mapping[old_id] for old_id in df[col].tolist()]

def build_iter_data(df, test_interactions, user_feature_cols, item_feature_cols):
    """
    :param df: user item interaction data
    :param x: number most recent transactions per user for test set
    :return: data iterators to feed model
    """
    # Ensure users have at least 20 interactions
    df = df.groupby(["user"], as_index=False, group_keys=False).filter(lambda x: x.shape[0] >= 20)

    # Reset user and movie ids to incremental range (since we may have removed values)
    reset_id(df, cols=["user", "movie"])

    n_users, n_items = len(df.user.unique()), len(df.movie.unique())

    # Select latest n interactions per user as test set
    test_positives=df.groupby(["user"], as_index=False, group_keys=False)\
        .apply(lambda x: x.nlargest(test_interactions, ["time"]))
    assert test_positives.shape[0] == n_users*test_interactions

    # Let train set be everything remaining
    train_positives=df[~df.index.isin(test_positives.index.values)]
    assert test_positives.shape[0]+train_positives.shape[0]==df.shape[0]

    # Create sparse matrix of train and test interactions (1 is train interaction, 2 is test interaction, 0 is negative)
    row = np.concatenate((np.array(train_positives.user.values.tolist()), np.array(test_positives.user.values.tolist())), axis=0)
    col = np.concatenate((np.array(train_positives.movie.values.tolist()), np.array(test_positives.movie.values.tolist())), axis=0)
    data = np.concatenate((np.array([1]*train_positives.shape[0]), np.array([2]*test_positives.shape[0])), axis=0)
    sparse_interactions = sps.csr_matrix((data, (row, col)), shape=(n_users, n_items))

    # Build unique user/item features
    user_features = df.drop_duplicates(subset="user").sort_values("user")[user_feature_cols].values.tolist()
    item_features = df.drop_duplicates(subset="movie").sort_values("movie")[item_feature_cols].values.tolist()

    # Build validation iterator, sampling x negatives per test interaction
    val_iter = iterators.SparseNegativeSamplingDataIter(sparse_interactions, user_features, item_features,
                                                        negatives_per_interaction=args.test_negatives, negative_sample_label=3,
                                                        interaction_label=2, interaction_labels=[1,2,3], batch_size=args.batch_size,
                                                        shuffle=True)


    # Build training iterator, making sure negatives sampled are not in the test set
    train_iter = iterators.SparseNegativeSamplingDataIter(val_iter.sparse_interactions, user_features, item_features,
                                                          negatives_per_interaction=args.train_negatives, negative_sample_label=3,
                                                          interaction_label=1, interaction_labels=[1,2,3], batch_size=args.batch_size)
    return train_iter, val_iter, n_users, n_items, df

def build_recsys_symbol(iterator, num_users, num_items, num_embed, dropout, fc_layers):
    """
    Build mxnet model symbol from data characteristics and prespecified hyperparameters.
    :return:  MXNet symbol object
    """
    user_databatch_shape, item_databatch_shape, labelbatch_shape = iterator.provide_data[0][1], \
                                                                   iterator.provide_data[1][1], \
                                                                   iterator.provide_label[0][1]


    user_x = mx.sym.Variable(name="user_x")
    item_x = mx.sym.Variable(name="item_x")
    softmax_label = mx.sym.Variable(name="softmax_label")

    print("user input: ", user_x.infer_shape(user_x=user_databatch_shape)[1][0])
    print("item input: ", item_x.infer_shape(item_x=item_databatch_shape)[1][0])
    print("label input: ", softmax_label.infer_shape(softmax_label=labelbatch_shape)[1][0])

    user_embed_layer = mx.sym.Embedding(data=user_x, input_dim=num_users, output_dim=num_embed, name="user_embedding")
    item_embed_layer = mx.sym.Embedding(data=item_x, input_dim=num_items, output_dim=num_embed, name="item_embedding")
    print("user feature embedding: ", user_embed_layer.infer_shape(user_x=user_databatch_shape)[1][0])
    print("item feature embedding: ", item_embed_layer.infer_shape(item_x=item_databatch_shape)[1][0])

    latent_feature_layer = mx.sym.reshape(mx.sym.concat(*[user_embed_layer, item_embed_layer], dim=2), shape=(0,-1))
    print("input features embedding: ", latent_feature_layer.infer_shape(user_x=user_databatch_shape,
                                                                         item_x=item_databatch_shape)[1][0])

    for i, layer_size in enumerate(fc_layers, start=1):
        if i == 1:
            fc = mx.sym.FullyConnected(data=latent_feature_layer, num_hidden=layer_size, name="fully connected layer " + str(i))
        else:
            fc = mx.sym.FullyConnected(data=act, num_hidden=layer_size, name="fully connected layer " + str(i))
        act = mx.sym.relu(data=fc, name="activated layer " + str(i))
        drp  = mx.sym.Dropout(data=act, p=dropout, name="dropout layer " + str(i))
        print("\tfully connected layer : ", fc.infer_shape(user_x=user_databatch_shape, item_x=item_databatch_shape)[1][0])

    pred = mx.sym.FullyConnected(data=drp, num_hidden=2, name='pred')
    print("prediction shape: ", pred.infer_shape(user_x=user_databatch_shape, item_x=item_databatch_shape)[1][0])

    return mx.sym.SoftmaxOutput(data=pred, label=softmax_label)

def train(symbol, train_iter, val_iter):
    """
    :param symbol: model symbol graph
    :param train_iter: data iterator for training data
    :param valid_iter: data iterator for validation data
    :return: model to predict label from features
    """
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    module = mx.mod.Module(symbol, data_names=train_iter.data_names, label_names=train_iter.label_names, context=devs)
    module.fit(train_data=train_iter,
               #eval_data=val_iter,
               optimizer=args.optimizer,
               eval_metric=mx.metric.Accuracy(),
               optimizer_params={'learning_rate': args.lr},
               initializer=mx.initializer.Normal(sigma=0.01),
               num_epoch=args.num_epochs,
               epoch_end_callback=evaluate(module, val_iter, args.topk, args.test_interactions, num_users))

if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Setup dirs
    if args.clean_output_dir:
        pass
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Read interaction data into pandas dataframe
    # ToDo: Find joined data from download
    train_df = pd.read_csv(os.path.join(args.data, "ml-1m.train.rating"), sep="\t", names = ["user", "movie", "rating", "time"], header=None, index_col=False)
    test_df = pd.read_csv(os.path.join(args.data, "ml-1m.test.rating"), sep="\t", names=["user", "movie", "rating", "time"], header=None, index_col=False)
    df=train_df.append(test_df)
    df.reset_index(inplace=True, drop=True)

    # Build data iterators
    train_iter, val_iter, num_users, num_items, df = build_iter_data(df, test_interactions=args.test_interactions,
                                                                 user_feature_cols=["user"], item_feature_cols=["movie"])

    # Build model symbol
    model_symbol = build_recsys_symbol(train_iter, num_users, num_items, args.num_embed, args.dropout, args.fc_layers)

    # Train the model
    train(model_symbol, train_iter, val_iter)