import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from random import random

import lightgbm as lgb
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn import datasets, ensemble, linear_model, metrics, svm
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from feature_loader import create_cv, prepare_data
from label_dict import label_dict, label_index
from utils import AverageMeter


def method_svm(random_state):
    params = {'kernel': 'rbf', 'random_state': random_state, 'verbose': 0,
              'probability': True, 'C': 3}
    return svm.SVC(**params), {'C': range(1, 30, 2)}


def method_lgbm(random_state):
    params = {'learning_rate': 0.01, 'n_jobs': 10, 'n_estimators': 3000, 'random_state': random_state, 'verbose': -1,
              'device': 'cpu', 'subsample': 0.5, 'feature_fraction': 0.01, 'lambda_l2': 0.1, 'max_depth': 1, 'min_data_in_leaf': 20}

    return lgb.LGBMClassifier(**params), {
        'learning_rate': sp_uniform(loc=0.001, scale=0.03),
        'subsample': sp_uniform(loc=0.5, scale=0.3),
        'max_depth': [1, 3, 7],
        'min_data_in_leaf': [1, 3, 7, 10, 20], }


def objective(optuna_trial):
    opt = parse_opts()

    random_state = 0

    X_train, y_train, X_val, y_val = prepare_data(opt, random_state)

    print(f'# train {len(X_train)}, # val {len(X_val)}')

    method = None
    if opt.method == 'svm':
        method = method_svm
    elif opt.method == 'lgbm':
        method = method_lgbm
    clf, cv_params = method(random_state)

    use_cv = False
    if use_cv:
        clf = create_cv(X_train, y_train, clf, cv_params)
    else:
        clf.fit(X_train, y_train)

    print()
    for i, c in label_dict.items():
        index = np.where(y_val == i)
        val_index = index[0]
        val_size = len(val_index)

        train_index = np.where(y_train == i)[0]
        train_size = len(train_index)
        if val_size:
            class_y = y_val[val_index]
            class_X = X_val[val_index]
            class_val_acc = metrics.accuracy_score(
                class_y, clf.predict(class_X))
            class_val_acc = format(class_val_acc, '.2f')
            print(
                f'{i}\t{c.ljust(10)}\t{class_val_acc}\t# t/v = {str(train_size).rjust(3)} / {str(val_size).rjust(3)}')

    predict_X_val = clf.predict(X_val)
    val_cm = metrics.confusion_matrix(y_val, predict_X_val)
    print(val_cm)

    import matplotlib.pyplot as plt
    plt.imshow(val_cm, cmap='binary', interpolation='None')
    val_cm_img_file_name = 'confusion_matrix.png'
    plt.savefig(val_cm_img_file_name)
    print(f'Export {val_cm_img_file_name}')

    import pandas as pd
    y_true = pd.Series(y_val)
    y_pred = pd.Series(predict_X_val)

    ct = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=[
                     'Predicted'], margins=True)
    print(ct)

    val_acc = metrics.accuracy_score(y_val, predict_X_val)
    print(f'val_acc = {val_acc}')

    val_micro_recall = metrics.recall_score(
        y_val, predict_X_val, average='micro')
    print(f'val_micro_recall = {val_micro_recall}')

    val_macro_f1 = metrics.f1_score(y_val, predict_X_val, average='macro')
    print(f'val_macro_f1 = {val_macro_f1}')

    val_micro_f1 = metrics.f1_score(y_val, predict_X_val, average='micro')
    print(f'val_micro_f1 = {val_micro_f1}')

    predict_prob_X_val = clf.predict_proba(X_val)

    top_n = 3
    top_n_prob = np.argsort(predict_prob_X_val, axis=1)[:, -top_n:]
    correct_count = 0
    for y, X_vals in zip(y_val, top_n_prob):
        if label_index[y] in X_vals:
            correct_count += 1
    val_loss = metrics.log_loss(y_val, predict_prob_X_val)
    print(f'val_loss = {val_loss}')
    print(f'top_{top_n}_val_acc = {correct_count / len(y_val)}')
    return val_loss


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--optuna_trials',
        default=0,
        type=int,
        help='Optuna trials')
    parser.add_argument(
        '--feature_path',
        default='/data/data_wedding/results/segment_to_feature.json',
        type=str,
        help='The path of segment_to_feature.json')
    parser.add_argument(
        '--method',
        default='',
        type=str,
        help='svm or lgbm')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    opt = parse_opts()
    print('=' * 100)
    print(f'OPTUNA_TRIALS = {opt.optuna_trials}')
    print('=' * 100)
    if opt.optuna_trials:
        study = optuna.create_study()
        study.optimize(objective, n_trials=opt.optuna_trials)
        print(study.best_params)
    else:
        objective(None)
