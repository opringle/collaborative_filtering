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
import pandas as pd

def TopKAccuracy(module, iterator, k, test_interactions_per_user, num_users):
    """
    groups by user
    orders by model output (predicted probability of interaction)
    selects top k per user
    takes average
    should be computed on all data at once
    :return:
    """
    user_ids, item_ids, labels, interaction_probs = [], [], [], []
    for pred, i_batch, batch in module.iter_predict(iterator):
        user_ids.extend(batch.data[0].asnumpy().flatten().tolist())
        item_ids.extend(batch.data[1].asnumpy().flatten().tolist())
        labels.extend(batch.label[0].asnumpy().tolist())
        interaction_probs.extend(pred[0].asnumpy()[:, 1].flatten().tolist())

    # Create pandas df
    df = pd.DataFrame(data={"item": item_ids, "pred_prob": interaction_probs, "label": labels}, index=user_ids)

    # Select k largest model predictions per user
    df = df.groupby(df.index).apply(lambda x: x.nlargest(k, ["pred_prob"]))
    print(df.head())

    # What percentage of the interactions would we have recommended?
    score = np.sum(df["label"]) / (test_interactions_per_user * num_users)

    print("Validation Top-{}-Accuracy: {}\n".format(k, score))