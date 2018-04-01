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
from ast import literal_eval

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Neural Collaborative Filtering Model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', nargs='?', default='../preprocessed_data/',
                        help='Input data path.')
parser.add_argument('--batch-size', type=int, default=500,
                    help='the number of training records in each minibatch')
parser.add_argument('--num-embed', type=int, default=50,
                    help='the user/item latent feature dimension')
parser.add_argument('--optimizer', type=str, default='Adam',
                    help='optimization algorithm to update model parameters with')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate for chosen optimizer')
parser.add_argument('--num-epochs', type=int, default=10,
                    help='how  many times to update the model parameters')
parser.add_argument('--disp-batches', type=int, default=5000,
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
                        fc_layers=[100, 50, 25]):
    """
    Build mxnet model symbol from data characteristics and prespecified hyperparameters.
    :return:  MXNet symbol object
    """
    user_databatch_shape, item_databatch_shape, labelbatch_shape = iterator.provide_data[0][1], iterator.provide_data[1][1], iterator.provide_label[0][1]

    user_x = mx.sym.Variable(name="user_x")
    item_x = mx.sym.Variable(name="item_x")
    softmax_label = mx.sym.Variable(name="softmax_label")
    print("user input: ", user_x.infer_shape(user_x=user_databatch_shape)[1][0])
    print("item input: ", item_x.infer_shape(item_x=item_databatch_shape)[1][0])
    print("label input: ", softmax_label.infer_shape(softmax_label=labelbatch_shape)[1][0])

    user_embed_layer = mx.sym.Embedding(data=user_x, input_dim=num_users, output_dim=num_embed, name="user_embedding")
    item_embed_layer = mx.sym.Embedding(data=item_x, input_dim=num_items, output_dim=num_embed, name="item_embedding")
    print("user feature embedding: ", user_embed_layer.infer_shape(user_x=user_databatch_shape)[1][0])
    print("item feature embedding: ", item_embed_layer.infer_shape(item_x = item_databatch_shape)[1][0])

    latent_feature_layer = mx.sym.concat(*[user_embed_layer, item_embed_layer], dim=1, name="combined_feature_embeddings")
    print("input features embedding: ", latent_feature_layer.infer_shape(user_x=user_databatch_shape, item_x=item_databatch_shape)[1][0])

    for i, layer_size in enumerate(fc_layers, start=1):
        if i == 1:
            fc = mx.sym.FullyConnected(data=latent_feature_layer, num_hidden=layer_size, name="fully connected layer " + str(i))
        else:
            fc = mx.sym.FullyConnected(data=act, num_hidden=layer_size, name="fully connected layer " + str(i))
        act = mx.sym.relu(data=fc, name="activated layer " + str(i))
        print("\tfully connected layer : ", fc.infer_shape(user_x=user_databatch_shape, item_x=item_databatch_shape)[1][0])

    label_layer = mx.sym.FullyConnected(data=act, num_hidden=1, name='label layer')
    loss_layer = mx.symbol.SoftmaxOutput(data=label_layer, label=softmax_label, name="loss layer")
    print("output layer : ", label_layer.infer_shape(user_x=user_databatch_shape, item_x = item_databatch_shape)[1][0])
    print("prediction layer : ", loss_layer.infer_shape(user_x=user_databatch_shape, item_x = item_databatch_shape, softmax_label=labelbatch_shape)[1][0])

    return loss_layer

def save_model(output_dir):
    """
    Save model files
    :return:
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    return mx.callback.do_checkpoint(prefix="checkpoint/checkpoint", period=args.save_period)

def train(symbol, train_iter):
    """
    :param symbol: model symbol graph
    :param train_iter: data iterator for training data
    :param valid_iter: data iterator for validation data
    :return: model to predict label from features
    """
    feature_name, label_name = train_iter.provide_data[0][0], train_iter.provide_label[0][0]
    module = mx.mod.Module(symbol, data_names=("user_x", "item_x"), label_names=("softmax_label", ), context=mx.cpu())
    module.fit(train_data=train_iter,
               eval_metric='loss',
               optimizer=args.optimizer,
               optimizer_params={'learning_rate': args.lr},
               initializer=mx.initializer.Uniform(0.1),
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
    user_input = np.load(os.path.join(args.path, "user_input.np"))
    item_input = np.load(os.path.join(args.path, "item_input.np"))
    labels = np.load(os.path.join(args.path, "labels.np"))
    with open(os.path.join(args.path, "users_items.txt"), "r") as f:
        num_users, num_items = literal_eval(f.readlines()[0])

    print("records in training set: {0}".format(user_input.shape[0]))

    #  build model data iterator
    train_iter = mx.io.NDArrayIter(data={'user_x':user_input, 'item_x':item_input}, label=labels, batch_size=args.batch_size)

    # build model symbol
    model_symbol = build_recsys_symbol(train_iter, num_users, num_items, args.num_embed)

    # train the model
    train(model_symbol, train_iter)