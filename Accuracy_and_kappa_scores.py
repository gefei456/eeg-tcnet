# *----------------------------------------------------------------------------*
# * Copyright (C) 2020 ETH Zurich, Switzerland                                 *
# * SPDX-License-Identifier: Apache-2.0                                        *
# *                                                                            *
# * Licensed under the Apache License, Version 2.0 (the "License");            *
# * you may not use this file except in compliance with the License.           *
# * You may obtain a copy of the License at                                    *
# *                                                                            *
# * http://www.apache.org/licenses/LICENSE-2.0                                 *
# *                                                                            *
# * Unless required by applicable law or agreed to in writing, software        *
# * distributed under the License is distributed on an "AS IS" BASIS,          *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# * See the License for the specific language governing permissions and        *
# * limitations under the License.                                             *
# *                                                                            *
# * Author:  Thorir Mar Ingolfsson                                             *
# *----------------------------------------------------------------------------*

# !/usr/bin/env python3

import numpy as np
import scipy.io as sio
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import load_model
import os
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from joblib import dump, load
import pickle
from sklearn.metrics import cohen_kappa_score
from utils.data_loading import prepare_features


def build_model(path):
    model = load_model(path)
    # model = load_model(path +'best.h5')
    for l in model.layers:
        l.trainable = False
    lr = 0.001
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
    return model


class Scaler(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self):
        self.scalers = {}
        for j in range(22):
            self.scalers[j] = StandardScaler()

    # Return self nothing else to do here
    def fit(self, X, y=None):
        for j in range(22):
            self.scalers[j].fit(X[:, 0, j, :])
        return self

        # Method that describes what we need this transformer to do

    def transform(self, X, y=None):
        for j in range(22):
            X[:, 0, j, :] = self.scalers[j].transform(X[:, 0, j, :])
        return X

stand = [True,True,True,True,True,True,True,True,True]
for i in range(9):
    if not(os.path.exists('models/EEG-TCNet/S{:}/pipeline_fixed.h5'.format(i+1))):
        print('Making Pipeline for Subject {:}'.format(i+1))
        path_for_model = 'models/EEG-TCNet/S{:}/model_fixed.h5'.format(i+1)
        clf = KerasClassifier(build_fn = build_model, path = path_for_model)
        if(stand[i]):
            pipe = make_pipeline(Scaler(),clf)
        else:
            pipe = make_pipeline(clf)
        data_path = 'data/'
        path = data_path+'s{:}/'.format(i+1)
        X_train,_,y_train_onehot,X_test,_,y_test_onehot = prepare_features(path,i,False)
        pipe.fit(X_train,y_train_onehot)
        X_train,_,y_train_onehot,X_test,_,y_test_onehot = prepare_features(path,i,False)
        y_pred = pipe.predict(X_test)
        dump(pipe, 'models/EEG-TCNet/S{:}/pipeline_fixed.h5'.format(i+1))
    else:
        print('Pipeline already exists for Subject {:}'.format(i+1))
print('Done!')

stand = [True,False,True,True,True,True,True,True,True]
for i in range(9):
    if not(os.path.exists('models/EEG-TCNet/S{:}/pipeline.h5'.format(i+1))):
        print('Making Pipeline for Subject {:}'.format(i+1))
        path_for_model = 'models/EEG-TCNet/S{:}/model.h5'.format(i+1)
        clf = KerasClassifier(build_fn = build_model,path=path_for_model)
        if(stand[i]):
            pipe = make_pipeline(Scaler(),clf)
        else:
            pipe = make_pipeline(clf)
        data_path = 'data/'
        path = data_path+'s{:}/'.format(i+1)
        X_train,_,y_train_onehot,X_test,_,y_test_onehot = prepare_features(path,i,False)
        pipe.fit(X_train,y_train_onehot)
        X_train,_,y_train_onehot,X_test,_,y_test_onehot = prepare_features(path,i,False)
        y_pred = pipe.predict(X_test)
        dump(pipe, 'models/EEG-TCNet/S{:}/pipeline.h5'.format(i+1))
    else:
        print('Pipeline already exists for Subject {:}'.format(i+1))
print('Done!')


for i in range(9):
    clf = load('models/EEG-TCNet/S{:}/pipeline_fixed.h5'.format(i+1))
    data_path = 'data/'
    path = data_path+'s{:}/'.format(i+1)
    X_train,_,y_train_onehot,X_test,_,y_test_onehot = prepare_features(path,i,False)
    y_pred = clf.predict(X_test)
    acc_score = accuracy_score(y_pred,np.argmax(y_test_onehot,axis=1))
    kappa_score = cohen_kappa_score(y_pred,np.argmax(y_test_onehot,axis=1))
    print('For Subject: {:}, Accuracy: {:}, Kappa: {:}.'.format(i+1,acc_score*100, kappa_score))

for i in range(9):
    clf = load('models/EEG-TCNet/S{:}/pipeline.h5'.format(i+1))
    data_path = 'data/'
    path = data_path+'s{:}/'.format(i+1)
    X_train,_,y_train_onehot,X_test,_,y_test_onehot = prepare_features(path,i,False)
    y_pred = clf.predict(X_test)
    acc_score = accuracy_score(y_pred,np.argmax(y_test_onehot,axis=1))
    kappa_score = cohen_kappa_score(y_pred,np.argmax(y_test_onehot,axis=1))
    print('For Subject: {:}, Accuracy: {:}, Kappa: {:}.'.format(i+1,acc_score*100, kappa_score))

