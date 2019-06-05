rm -rf /data/data_wedding/jpg
mkdir /data/data_wedding/jpg
python utils/video_jpg_wedding.py /data/data_wedding/train /data/data_wedding/jpg
python utils/n_frames_kinetics.py /data/data_wedding/jpg
python utils/wedding_json.py /data/data_wedding/labels/train.txt  /data/data_wedding/labels/val.txt /data/data_wedding/labels/test.txt /data/data_wedding/wedding.json
python feature_dump.py --root_path /data/data_wedding  --video_path jpg --annotation_path wedding.json --result_path results --dataset wedding --n_classes 400 --n_finetune_classes 15 --pretrain_path models/resnet-34-kinetics.pth --ft_begin_index 4 --model resnet --model_depth 34 --resnet_shortcut A --batch_size 64 --n_threads 0 --checkpoint 100 --n_epochs 200
python classification.py --feature_path /data/data_wedding/results/segment_to_feature_34.json --optuna_trials 0 --method svm