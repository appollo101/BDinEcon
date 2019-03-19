# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 22:47:19 2019

@author: mgu3, folked from @LastRocky
"""

from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, Conv2D, MaxPooling2D, Flatten
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import lightgbm as lgb
import xgboost as xgb

######################### Log Loss Metric ####################################
def cal_logloss(y_true, y_pred):
    return log_loss(y_true, y_pred)

######################### Neural Networks model ##############################
def nn_model(paras, data):
    x_tr, y_tr, x_val, y_val = data['x_tr'], data['y_tr'], data['x_val'], data['y_val']
    input_nodes = x_tr.shape[1]
    layer_1_nodes = paras['nn_l1']
    layer_2_nodes = paras['nn_l2']
    batch = paras['batch']
    number_of_epochs = paras['epochs']
    dropout_rate = paras['dp']
    nn_model = Sequential()

    # The input layer and the first hidden layer
    nn_model.add(Dense(activation="relu", input_dim=input_nodes, units=
                       layer_1_nodes, kernel_initializer="lecun_normal",
                         kernel_regularizer=regularizers.l2(0.01)))
    # The second hidden layer
    nn_model.add(Dropout(dropout_rate))
    nn_model.add(Dense(activation="relu", input_dim=layer_1_nodes, units=
                       layer_2_nodes, kernel_initializer="lecun_normal",
                         kernel_regularizer=regularizers.l2(0.01)))
    nn_model.add(Dropout(dropout_rate))

    nn_model.add(Dense(activation="sigmoid", units=1, kernel_initializer=
                       "lecun_normal"))
    # Compile the ANN
    nn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', )
    filepath="poverty_weights.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]
    history_ts = nn_model.fit(x_tr, y_tr,batch_size = batch, epochs = number_of_epochs, validation_data=(x_val,y_val),
    callbacks=callbacks_list, verbose=0)
    nn_model.load_weights("poverty_weights.hdf5")
    y_pred_val = nn_model.predict(x_val).ravel()
    y_pred_test = nn_model.predict(data['x_test'].values).ravel()

    return y_pred_val, y_pred_test

################# Light Gradient Boosting Decision Trees #####################
def lgb_model(paras, data):
    x_tr, y_tr, x_val, y_val = data['x_tr'], data['y_tr'], data['x_val'], data['y_val']
    lgb_params = {
        'task' : 'train',
        'boosting_type' : 'gbdt',
        'objective' : 'binary',
        'metric' : {'binary_logloss'},
        'max_depth': paras['max_depth'],
        'is_training_metric': False,
        'learning_rate' : paras['lr'],
        'feature_fraction' : paras['feature_fraction'],
        'min_sum_hessian_in_leaf': paras['hess'],
        'verbose': -1,
    }
    lgb_train = lgb.Dataset(x_tr, y_tr, feature_name=paras['col_names'])
    lgb_val = lgb.Dataset(x_val, y_val, feature_name=paras['col_names'],
                          reference=lgb_train)
    watchlist = lgb_val
    lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=20000,
                          valid_sets=watchlist, early_stopping_rounds=1000,
                          verbose_eval=paras['verbos_'])
    y_pred_val = lgb_model.predict(x_val, num_iteration=
                                   lgb_model.best_iteration).ravel()
    y_pred_test = lgb_model.predict(data['x_test'], num_iteration=
                                    lgb_model.best_iteration).ravel()
    return y_pred_val, y_pred_test

##################### eXtreme Gradient Boosting Model ########################
def xgb_model(paras, data):
    x_tr, y_tr, x_val, y_val = data['x_tr'], data['y_tr'], data['x_val'], data['y_val']

    xgb_params = {
        'eta': paras['eta'],
        'max_depth': paras['max_depth'],
        'subsample': paras['subsample'],
        'colsample_bytree': paras['colsample_by_tree'],
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        #'base_socre': 0.2,
        'seed': 123,
        'silent': 1,
    }
    dtrain = xgb.DMatrix(x_tr, label=y_tr, feature_names=paras['col_names'])
    dval= xgb.DMatrix(x_val, label=y_val, feature_names=paras['col_names'])

    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=20000, evals=[(dtrain, 'train'), (dval, 'val')],
                          early_stopping_rounds=1000, verbose_eval=paras['verbos_'])
    y_pred_val = xgb_model.predict(dval, ntree_limit=xgb_model.best_ntree_limit)
    y_pred_test = xgb_model.predict(xgb.DMatrix(data['x_test'].values, feature_names=paras['col_names']),
                                      ntree_limit=xgb_model.best_ntree_limit)
    return y_pred_val, y_pred_test


def train_model(features, labels, paras, test_ = None):
    losses = []
    preds = []

    skf = StratifiedKFold(n_splits=paras['splits'], random_state=123, shuffle=True)
    for i, (train_index, val_index) in enumerate(skf.split(features, labels)):
        print("Iteration: ", i, " Current time: ", str(datetime.now()))
        x_tr, x_val = features.values[train_index], features.values[val_index]
        y_tr, y_val = labels[train_index], labels[val_index]
        data = {'x_tr': x_tr, 'x_val': x_val, 'y_tr': y_tr, 'y_val': y_val, 'x_test': test_}

        # lgb model
        y_pred_val_lgb, y_pred_test_lgb = lgb_model(paras['lgb'], data)
        print('*********************************Logloss from lgb is: {}\n'.format(cal_logloss(y_val, y_pred_val_lgb)))
        print('lgb, max: {}, min: {}, mean: {}'.format(np.max(y_pred_test_lgb), np.min(y_pred_test_lgb), np.mean(y_pred_test_lgb)))

        # xgb model
        y_pred_val_xgb, y_pred_test_xgb = xgb_model(paras['xgb'], data)
        print('*********************************Logloss from xgb is: {}\n'.format(cal_logloss(y_val, y_pred_val_xgb)))
        print('xgb, max: {}, min: {}, mean: {}'.format(np.max(y_pred_test_xgb), np.min(y_pred_test_xgb), np.mean(y_pred_test_xgb)))

        if paras['use_nn']:
            # neural network model
            y_pred_val_nn, y_pred_test_nn = nn_model(paras['nn'], data)
            print('*********************Logloss from neural network is: {}\n'.format(cal_logloss(y_val, y_pred_val_nn)))
            print('nn, max: {}, min: {}, mean: {}'.format(np.max(y_pred_test_nn), np.min(y_pred_test_nn), np.mean(y_pred_test_nn)))
            y_val_xgb_lgb_nn = y_pred_val_lgb * paras['w_lgb'] + y_pred_val_xgb * paras['w_xgb'] + y_pred_val_nn * paras['w_nn']
            loss = cal_logloss(y_val, y_val_xgb_lgb_nn)
            y_pred_test_final = y_pred_test_lgb * paras['w_lgb'] + y_pred_test_xgb * paras['w_xgb'] + y_pred_test_nn * paras['w_nn']
            y_val_xgb_lgb = (y_pred_val_lgb * paras['w_lgb'] + y_pred_val_xgb * paras['w_xgb'])/(paras['w_xgb'] + paras['w_lgb'])
            loss_xl = cal_logloss(y_val, y_val_xgb_lgb)
            print('**********Logloss from combination of xgb and lgb is:     {}\n'.format(loss_xl))
            print('**********Logloss from combination of xgb, lgb and nn is: {}\n'.format(loss))
            print('max for lgb: {}, xgb: {}, nn: {}'.format(np.max(y_pred_test_lgb), np.max(y_pred_test_xgb), np.max(y_pred_test_nn)))
        else:
            y_val_lgb_xgb = (y_pred_val_lgb * paras['w_lgb'] + y_pred_val_xgb * paras['w_xgb'])/(paras['w_xgb'] + paras['w_lgb'])
            loss = cal_logloss(y_val, y_val_lgb_xgb)
            y_pred_test_final = (y_pred_test_lgb * paras['w_lgb'] + y_pred_test_xgb * paras['w_xgb'])/(paras['w_xgb'] + paras['w_lgb'])
            print('**********Logloss from combination of xgb and lgb is: {}\n'.format(loss))
        losses.append(loss)
        print('Max value for predicted test data: {} and mean value: {}\n'.format(np.max(y_pred_test_final), np.mean(y_pred_test_final)))
        preds.append(y_pred_test_final)
    m_loss = np.mean(losses)
    print(losses, " mean log loss: ", m_loss)
    p_a = np.mean(preds, axis=0)

    return p_a, m_loss

# load A_hhold training data
a_train = pd.read_csv('../A_hhold_train.csv', index_col='id')
# load A_hhold test data
a_test_o = pd.read_csv('../A_hhold_test.csv', index_col='id')
print('Shape of hhold data A, Train and test: \n', a_train.shape, a_test_o.shape)
print('Mean value of "poor" for A: \n',a_train.poor.mean())

num_cols_a = [col for col in a_train.columns if a_train[col].dtype in ['int64', 'float64'] and col not in ['id'] and '_mean' not in col]

a_hhold = pd.concat([a_train.drop(['country'], axis=1), a_test_o], axis=0)

######################## Standardize features ################################
def standardize(df, numeric_only=True):
    numeric = df.select_dtypes(include=['int64', 'float64'])
# subtracy mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()
    return df

######################## Pre-processing data #################################
def pre_process_data(df, nn=False):
    print("Input shape:\t{}".format(df.shape))
    df = standardize(df)
#    if nn:
    poor_ = df.poor.values
    df = pd.get_dummies(df.drop('poor', axis=1), drop_first=True)
    df['poor'] = poor_.astype('bool')
    all_nan_cols = []
    for col in df.columns:
        if df[col].isnull().sum() == df.shape[0]:
            all_nan_cols.append(col)
    df.drop(all_nan_cols, axis=1, inplace=True)
    df.fillna(df.median(), inplace=True)
#    else:
#        df = encode_cat(df)
    print('Final shape {}'.format(df.shape))
    return df

a_hhold_p = pre_process_data(a_hhold, nn=True)
print()

aX_train = a_hhold_p.drop('poor', axis=1)[:a_train.shape[0]]
ay_train = np.ravel(a_hhold_p.poor)[:a_train.shape[0]]
a_test = a_hhold_p.drop('poor', axis=1)[a_train.shape[0]:]


########################### Start training ###################################

paras_a = {
    'splits': 20,
    'lgb': {
        'max_depth': 4,
        'lr': 0.01,
        'hess': 3.,
        'feature_fraction': 0.07,
        'verbos_': 1000,
        'col_names': aX_train.columns.tolist(),
    },
    'xgb': {
        'eta': 0.01,
        'max_depth': 4,
        'subsample': 0.75,
        'colsample_by_tree': 0.07,
        'verbos_': 1000,
        'col_names': aX_train.columns.tolist(),
    },
    'use_nn': True,
    'nn': {
        'nn_l1': 300,
        'nn_l2': 300,
        'epochs': 75,
        'batch': 64,
        'dp': 0.,
    },
    'w_xgb': 0.45,
    'w_lgb': 0.25,
    'w_nn': 0.3,
}

a_preds, a_loss = train_model(aX_train, ay_train, paras_a, test_ = a_test)

def make_country_sub(preds, test_feat, country):

    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds,  # proba p=1
                               columns=['poor'],
                               index=test_feat.index)
    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]

# convert preds to data frames
a_sub = make_country_sub(a_preds, a_test, 'A')

print(a_sub.poor.agg(['max', 'min', 'mean', 'median']))

print('Local logloss cross validation score: {}'.format(a_loss))

