from __future__ import division, print_function

import json
import os
import sys

from video_jpg_wedding import VIDAUG_BATCHES


def convert_csv_to_dict(csv_path, subset):
    keys = []
    key_labels = []
    with open(csv_path, 'r') as fp:
        lines = fp.read().splitlines()
        for l in lines:
            tokens = list(filter(None, l.split(',')))
            src_file = tokens[0]
            class_name = tokens[1].strip()
            folder, file_name_and_ext = os.path.split(src_file)
            name, ext = os.path.splitext(file_name_and_ext)

            key = f"{folder}__{name}___{subset}_{class_name}"
            keys.append(key)
            if subset != 'testing':
                key_labels.append(class_name)

            for batch_index in range(VIDAUG_BATCHES):
                key = f"{folder}_vidaug_{batch_index}__{name}___{subset}_{class_name}"
                keys.append(key)
                if subset != 'testing':
                    key_labels.append(class_name)

    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        if subset != 'testing':
            label = key_labels[i]
            database[key]['annotations'] = {'label': label}
        else:
            database[key]['annotations'] = {}

    return database


def load_labels(train_csv_path, val_csv_path):
    class_names = set()
    for path in [train_csv_path, val_csv_path]:
        with open(path, 'r') as fp:
            lines = fp.read().splitlines()
            for l in lines:
                tokens = list(filter(None, l.split(',')))
                src_file = tokens[0]
                class_name = tokens[1].strip()
                class_names.add(class_name)
    class_names = list(class_names)
    class_names.sort()
    return class_names


def convert_kinetics_csv_to_activitynet_json(train_csv_path, val_csv_path, test_csv_path, dst_json_path):
    labels = load_labels(train_csv_path, val_csv_path)
    train_database = convert_csv_to_dict(train_csv_path, 'training')
    val_database = convert_csv_to_dict(val_csv_path, 'validation')
    test_database = convert_csv_to_dict(test_csv_path, 'testing')

    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)
    dst_data['database'].update(test_database)
    print(f"{len(labels)} classes.")

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == "__main__":
    train_csv_path = sys.argv[1]
    val_csv_path = sys.argv[2]
    test_csv_path = sys.argv[3]
    dst_json_path = sys.argv[4]

    convert_kinetics_csv_to_activitynet_json(
        train_csv_path, val_csv_path, test_csv_path, dst_json_path)
