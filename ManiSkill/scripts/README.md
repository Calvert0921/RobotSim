# Other scripts
This folder contains scripts which should be used in other repos: *lerobot* and *maniskill-data-vla*
## lift_cubes.py
Env creater, contains five cubes with different colours
## convert_to_lerobot.py
Convert raw data to lerobot data format
### Usage:
```
python convert_to_lerobot.py --data_path $path$ --push_to_hub 
```
## combine_datasets.py
Combine two seperate Hugging Face datasets, address the issue when the storage is limited
### Usage:
```
python combine_datasets.py \
  --src dataset1 \
  --src dataset2 \
  --dest_repo combined_dataset \
  --dest_root ~/.cache/huggingface/lerobot_merged \
  --copy_videos \
  --recompute_info \
  --push \
  --tag
```
