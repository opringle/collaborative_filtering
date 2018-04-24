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
import scipy
import mxnet as mx
import numpy as np
import argparse
import logging
import os
import metrics
from ast import literal_eval

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
parser.add_argument('--lr', type=float, default=0.0001,
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

def negatives_per_user():
    """
    Randomly selects x negative values for each user in a sparse interaction csr matrix
    :return:
    """

def build_train_test(df, test_interactions):
    """
    :param df: user item interaction data
    :param x: number most recent transactions per user
    :return: train & test pandas dataframes
    """
    #ToDo: Ensure we only have each user and movie once??
    df = df[:100]#.drop_duplicates(["user", "movie"])

    # Select latest n interactions per user as test set
    test_positives=df.groupby(["user"], as_index=False, group_keys=False)\
        .apply(lambda x: x.nlargest(test_interactions, ["time"]))

    # Let train set be everything remaining
    train_positives=df[~df.index.isin(test_positives.index.values)]

    # Build sparse representation of all data (positives and negatives)
    # ToDo: This is format of (row, col) data
    sparse_interactions=scipy.sparse.csr_matrix(test_positives.values)
    print("\nDataframe\n{}\nSparse Df\n{}\n".format(test_positives, sparse_interactions))

    # For each user randomly select x negatives
    train_negatives = [np.random.choice(x) for x in sparse_interactions.rows if x]
    test_negatives = [np.random.choice(x) for x in sparse_interactions.rows if x]

    # Build training negatives

    # Save to file

    # print(df.head())
    # print("\n", train_df.head(), "\n", test_df.head())
    # print(sparse_interactions[:10])
    # #print(train_negatives)
    # print(sparse_interactions.rows)



    # ToDo: We are training at serious scale here, therefore our iterators will read from a file line by line
    # ToDo: We want to store positive examples only, since if we store negatives we will have an enourmous dataset

    # When training can we just randomly sample negatives
    # ToDo: Or do we? We could just store all data and rely on distributed computing.

    # ToDo: We want to get the most real evaluation of our model, so we understand what happens when it is used
    # ToDo: Therefore, we want all test data? If we had 5 possible ranking spots, and ranked all items for the user, what % of recommendations were purchased by the user?


    # ToDo: Iterator needs to know which interactions are test, which are train, which are the test samples, how many train samples to select
    # ToDo: Don't want to store values for negatives as then we run out of memory
    # ToDo: Instead, lets keep track of three things. Train interactions, Test interactions & Train Negatives

    # ToDo: This is because there are less train negatives per user than train  positives often (maybe? why?)




    return train_df, test_df


def build_recsys_symbol(iterator,
                        num_users,
                        num_items,
                        num_embed,
                        dropout,
                        fc_layers):
    """
    Build mxnet model symbol from data characteristics and prespecified hyperparameters.
    :return:  MXNet symbol object
    """
    user_databatch_shape, item_databatch_shape, labelbatch_shape = iterator.provide_data[0][1], \
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

    latent_feature_layer = mx.sym.concat(*[user_embed_layer, item_embed_layer], dim=1,
                                         name="combined_feature_embeddings")
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

    pred = mx.sym.reshape(mx.sym.sigmoid(mx.sym.FullyConnected(data=drp, num_hidden=1)), shape=(0, ), name='pred')
    log_loss = - (softmax_label * mx.sym.log(pred)) - ((1-softmax_label) * mx.sym.log(1-pred))
    loss_grad = mx.sym.make_loss(log_loss)
    print("prediction layer : ", pred.infer_shape(user_x=user_databatch_shape, item_x = item_databatch_shape)[1][0])

    return mx.sym.Group([mx.sym.BlockGrad(pred, name="pred"), loss_grad])

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

    feature_name, label_name = train_iter.provide_data[0][0], train_iter.provide_label[0][0]
    module = mx.mod.Module(symbol, data_names=("user_x", "item_x"), label_names=("softmax_label", ), context=mx.cpu())
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
    #print("\ndf head: \n{}\nshape of df: {}. Max user id: {}\n".format(df.head(),df.shape, max(df.user)))

    # Test set is latest x interactions per user
    train_df, test_df = build_train_test(df, test_interactions=1)


    #
    # # clean the output directory
    # if args.clean_output_dir == True:
    #     filelist = [f for f in os.listdir(args.output_dir)]
    #     for f in filelist:
    #         os.remove(os.path.join(args.output_dir, f))
    #
    # # # read in preprocessed data
    # # X_train_user = np.load(os.path.join(args.path, "X_train_user.npy"))
    # # X_train_item = np.load(os.path.join(args.path, "X_train_item.npy"))
    # # Y_train = np.load(os.path.join(args.path, "Y_train.npy"))
    # # X_test_user = np.load(os.path.join(args.path, "X_test_user.npy"))
    # # X_test_item = np.load(os.path.join(args.path, "X_test_item.npy"))
    # # Y_test = np.load(os.path.join(args.path, "Y_test.npy"))
    # # with open(os.path.join(args.path, "users_items.txt"), "r") as f:
    # #     num_users, num_items = literal_eval(f.readlines()[0])
    # # print("records in training set: {0}".format(len(X_train_user)))
    #
    # #  build model data iterator
    #
    # # Build data iterators
    # train_iter = mx.io.NDArrayIter(data={'user_x':X_train_user, 'item_x':X_train_item}, label=Y_train,
    #                                batch_size=args.batch_size, shuffle=True)
    # val_iter = mx.io.NDArrayIter(data={'user_x': X_test_user, 'item_x': X_test_item}, label=Y_test,
    #                              batch_size=args.batch_size, shuffle=True)
    #
    # # Build model symbol
    # model_symbol = build_recsys_symbol(train_iter, num_users, num_items, args.num_embed, args.dropout, args.fc_layers)
    # print(model_symbol.list_outputs())
    #
    # # Train the model
    # train(model_symbol, train_iter, val_iter, metric=metrics.PreRecF1())