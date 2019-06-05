
import json
import os
from collections import defaultdict

import numpy as np
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, SVMSMOTE, RandomOverSampler
from scipy.sparse import coo_matrix
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.utils import shuffle

from label_dict import label_dict, label_index

VIDAUG_BATCHES = 0


def prepare_rgb_data(opt, folders):
    file_names, X, y = [], [], []
    segment_to_feature = {}
    with open(opt.feature_path, 'r') as f:
        segment_to_feature = json.load(f)

    file_to_segments = defaultdict(list)
    for k, v in segment_to_feature.items():
        file_name, index = k.rsplit('_', 1)
        file_name = file_name.split('/')[1]
        folder = file_name.split('__')[0]
        if folder in folders:
            file_to_segments[file_name].append(k)

    for k, v in file_to_segments.items():
        features = [segment_to_feature[w]['feature'] for w in v]
        feature = np.average(features, axis=0)

        pair = segment_to_feature[v[0]]
        label = pair['label']
        if not label in []:
            file_names.append(os.path.join(label, k))
            X.append(feature)
            y.append(label)
    return file_names, X, y


def prepare_pose_data(opt, folders):
    file_names, X, y = [], [], []
    segment_to_feature = {}
    with open('/data/data_wedding/file_to_pose.json', 'r') as f:
        segment_to_feature = json.load(f)

    count = 0
    for k, v in segment_to_feature.items():
        count += 1
        folder, file_name = k.split('__')
        if folder in folders:
            label = v[0]['label']

            if not label in ['c11']:
                keypoints_in_frames = []
                for frame in v:
                    persons = frame['outputs']

                    couple_or_single = sorted(
                        persons, key=lambda k: k['score'], reverse=True)[:2]
                    # MS COCO annotation order:
                    # 0: nose	   		1: l eye		2: r eye	3: l ear	4: r ear
                    # 5: l shoulder	6: r shoulder	7: l elbow	8: r elbow
                    # 9: l wrist		10: r wrist		11: l hip	12: r hip	13: l knee
                    # 14: r knee		15: l ankle		16: r ankle

                    # 17 x 3 = 51
                    keypoints = [c['keypoints'] for c in couple_or_single]

                    if keypoints:
                        if len(keypoints) == 1:
                            keypoints_in_frames.append(
                                keypoints[0]+keypoints[0])
                        else:
                            keypoints_in_frames.append(
                                keypoints[0]+keypoints[1])
                if keypoints_in_frames:
                    abs_diffs = []
                    for i in range(len(keypoints_in_frames) - 1):
                        frame1 = keypoints_in_frames[i]
                        frame2 = keypoints_in_frames[i+1]
                        abs_diff = [abs(f2 - f1)
                                    for f2, f1 in zip(frame2, frame1)]
                        abs_diffs.append(abs_diff)
                    if abs_diffs:
                        file_names.append(os.path.join(label, k))
                        feature = np.average(
                            abs_diffs, axis=0) + np.average(keypoints_in_frames, axis=0)
                        X.append(feature)
                        y.append(label)
    return file_names, X, y


def create_cv(X_train, y_train, estimator, param_distributions):

    cv = StratifiedKFold(n_splits=5)
    gscv = RandomizedSearchCV(
        estimator=estimator, param_distributions=param_distributions,
        n_iter=20,
        n_jobs=3,
        scoring='f1_micro',
        cv=cv,
        refit=True,
        random_state=0,
        verbose=10)
    gscv.fit(X_train, y_train)

    print(gscv.best_params_, gscv.best_score_)
    return gscv.best_estimator_


def prepare_data(opt, random_state):
    data_folders = ['3_14_yang_mov',
                    '3_24_guo_mov',
                    '3_8_weng',
                    '4_18_zheng',
                    'caiwei',
                    'daidai_mov',
                    'xia',
                    'shi_mov']

    vidaug_interfix = '_vidaug_'
    aug_folders = []
    for batch_index in range(VIDAUG_BATCHES):
        aug_folders += [f'{f}{vidaug_interfix}{batch_index}' for f in data_folders]

    file_names,  X, y = prepare_rgb_data(opt, data_folders)
    X = [(f, x) for f, x in zip(file_names, X)]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, shuffle=True, random_state=random_state, stratify=y)

    X_train_file_names = [f for (f, x) in X_train]
    X_train = [x for (f, x) in X_train]
    X_val = [x for (f, x) in X_val]

    file_names, X_aug, y_aug = prepare_rgb_data(opt, aug_folders)
    for f, xa, ya in zip(file_names,  X_aug, y_aug):
        for batch_index in range(VIDAUG_BATCHES):
            f = f.replace(f'{vidaug_interfix}{batch_index}', '')
            if f in X_train_file_names:
                X_train.append(xa)
                y_train.append(ya)

    data_size = len(X_train) + len(X_val)
    print(f'Total data size: {data_size}')

    # sampler should be disabled when running cv
    sampler = RandomOverSampler
    X_train, y_train = sampler(
        random_state=random_state).fit_sample(X_train, y_train)

    X_train_sparse = coo_matrix(X_train)
    X_train, _, y_train = shuffle(
        X_train, X_train_sparse, y_train, random_state=random_state)
    return np.asarray(X_train), np.asarray(y_train), np.asarray(X_val), np.asarray(y_val)
