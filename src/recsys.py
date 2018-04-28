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
import os
import iterators
import time

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Neural Collaborative Filtering Model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', nargs='?', default='../preprocessed_data/',
                        help='Input data path.')
parser.add_argument('--batch-size', type=int, default=512,
                    help='the number of training records in each minibatch')
parser.add_argument('--num-embed', type=int, default=16,
                    help='the user/item latent feature dimension')
parser.add_argument('--fc-layers', type=list, default=[32, 16, 8],
                    help='list of hidden layer sizes')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='the user/item latent feature dimension')
parser.add_argument('--optimizer', type=str, default='Adam',
                    help='optimization algorithm to update model parameters with')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate for chosen optimizer')
parser.add_argument('--num-epochs', type=int, default=100,
                    help='how  many times to update the model parameters')
parser.add_argument('--disp-batches', type=int, default=10000,
                    help='show progress for every n batches')
parser.add_argument('--save-period', type=int, default=10,
                    help='save model every n epochs')
parser.add_argument('--output-dir', type=str, default='checkpoint',
                    help='directory to save model params/symbol to')
parser.add_argument('--clean-output-dir', type=bool, default=True,
                    help='delete previous output directory files')

def build_iter_data(df, test_interactions, user_feature_cols, item_feature_cols):
    """
    :param df: user item interaction data
    :param x: number most recent transactions per user for test set
    :return: data iterators to feed model
    """
    n_users, n_items = len(df.user.unique()), len(df.movie.unique())

    # Select latest n interactions per user as test set
    test_positives=df.groupby(["user"], as_index=False, group_keys=False)\
        .apply(lambda x: x.nlargest(test_interactions, ["time"]))

    # Let train set be everything remaining
    train_positives=df[~df.index.isin(test_positives.index.values)]

    # Create sparse matrix of train and test interactions (1 is train interaction, 2 is test interaction)
    row = np.concatenate((np.array(train_positives.user.values.tolist()), np.array(test_positives.user.values.tolist())), axis=0)
    col = np.concatenate((np.array(train_positives.movie.values.tolist()), np.array(test_positives.movie.values.tolist())), axis=0)
    data = np.concatenate((np.array([1]*train_positives.shape[0]), np.array([2]*test_positives.shape[0])), axis=0)
    sparse_interactions = sps.csr_matrix((data, (row, col)), shape=(n_users, n_items))

    # Build unique user/item features for iterator to read from
    user_features = df.drop_duplicates(subset="user").sort_values("user")[user_feature_cols].values.tolist()
    item_features = df.drop_duplicates(subset="movie").sort_values("movie")[item_feature_cols].values.tolist()

    # Build training iterator
    train_iter = iterators.SparseNegativeSamplingDataIter(sparse_interactions, user_features, item_features,
                                                          negatives_per_user=100, negative_sample_label=3,
                                                          interaction_label=1, batch_size=500)

    # Validation iterator uses sparse matrix from train iter to ensure train negatives are not resampled
    val_iter = iterators.SparseNegativeSamplingDataIter(train_iter.sparse_interactions, user_features, item_features,
                                                          negatives_per_user=4, negative_sample_label=3,
                                                          interaction_label=2, batch_size=500)

    return train_iter, val_iter, n_users, n_items

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

    # pred = mx.sym.reshape(mx.sym.sigmoid(mx.sym.FullyConnected(data=drp, num_hidden=1)), shape=(0, ), name='pred')
    # log_loss = - (softmax_label * mx.sym.log(pred)) - ((1-softmax_label) * mx.sym.log(1-pred))
    # loss_grad = mx.sym.make_loss(log_loss)
    # print("prediction layer : ", pred.infer_shape(user_x=user_databatch_shape, item_x = item_databatch_shape)[1][0])
    #
    # return mx.sym.Group([mx.sym.BlockGrad(pred, name="pred"), loss_grad])
    return mx.sym.SoftmaxOutput(data=drp, label=softmax_label)

def save_model(output_dir):
    """
    Save model files
    :return:
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    return mx.callback.do_checkpoint(prefix="checkpoint/checkpoint", period=args.save_period)

def train(symbol, train_iter, val_iter, metric):
    """
    :param symbol: model symbol graph
    :param train_iter: data iterator for training data
    :param valid_iter: data iterator for validation data
    :return: model to predict label from features
    """
    module = mx.mod.Module(symbol, data_names=train_iter.data_names, label_names=train_iter.label_names, context=mx.cpu())
    module.fit(train_data=train_iter,
               eval_data=val_iter,
               eval_metric=metric,
               optimizer=args.optimizer,
               optimizer_params={'learning_rate': args.lr},
               initializer=mx.initializer.Normal(sigma=0.01),
               num_epoch=args.num_epochs,
               batch_end_callback=mx.callback.Speedometer(args.batch_size, args.disp_batches),
               epoch_end_callback=save_model(args.output_dir))

if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Read interaction data into pandas dataframe
    train_df = pd.read_csv("../data/ml-1m.train.rating", sep="\t", names = ["user", "movie", "rating", "time"], header=None)
    test_df = pd.read_csv("../data/ml-1m.test.rating", sep="\t", names=["user", "movie", "rating", "time"], header=None)
    df=train_df.append(test_df)
    df.index.name="id"

    # Build data iterators
    train_iter, val_iter, num_users, num_items = build_iter_data(df, test_interactions=1, user_feature_cols=["user"],
                                                                 item_feature_cols=["movie"])

    # Build model symbol
    model_symbol = build_recsys_symbol(train_iter, num_users, num_items, args.num_embed, args.dropout, args.fc_layers)

    # Train the model
    train(model_symbol, train_iter, val_iter, metric=mx.metric.Accuracy())#output_names=["pred_output"]))

    #
    # # clean the output directory
    # if args.clean_output_dir == True:
    #     filelist = [f for f in os.listdir(args.output_dir)]
    #     for f in filelist:
    #         os.remove(os.path.join(args.output_dir, f))
    #
    # start = time.time()
    # for i, batch in enumerate(train_iter):
    #     if i ==0:
    #         print("batch: {}\n\n\nX user: {}\n\n\nX item: {}\n\n\nlabel: {}".format(i, batch.data[0].shape, batch.data[1].shape, batch.label[0].shape))
    # print("batch time per epoch: {}s".format(time.time()- start))