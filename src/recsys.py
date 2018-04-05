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
    # parse args
    args = parser.parse_args()

    # clean the output directory
    if args.clean_output_dir == True:
        filelist = [f for f in os.listdir(args.output_dir)]
        for f in filelist:
            os.remove(os.path.join(args.output_dir, f))

    # read in preprocessed data
    X_train_user = np.load(os.path.join(args.path, "X_train_user.npy"))
    X_train_item = np.load(os.path.join(args.path, "X_train_item.npy"))
    Y_train = np.load(os.path.join(args.path, "Y_train.npy"))
    X_test_user = np.load(os.path.join(args.path, "X_test_user.npy"))
    X_test_item = np.load(os.path.join(args.path, "X_test_item.npy"))
    Y_test = np.load(os.path.join(args.path, "Y_test.npy"))
    with open(os.path.join(args.path, "users_items.txt"), "r") as f:
        num_users, num_items = literal_eval(f.readlines()[0])
    print("records in training set: {0}".format(len(X_train_user)))

    #  build model data iterator
    train_iter = mx.io.NDArrayIter(data={'user_x':X_train_user, 'item_x':X_train_item}, label=Y_train,
                                   batch_size=args.batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(data={'user_x': X_test_user, 'item_x': X_test_item}, label=Y_test,
                                 batch_size=args.batch_size, shuffle=True)

    # build model symbol
    model_symbol = build_recsys_symbol(train_iter, num_users, num_items, args.num_embed, args.dropout, args.fc_layers)
    print(model_symbol.list_outputs())

    # get a custom metric
    metric = metrics.PreRecF1()

    # train the model
    train(model_symbol, train_iter, val_iter, metric)