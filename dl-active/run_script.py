import os
import sys
import argparse

## This script shows the commands that run on different terminals

# run on terminal 0: train the first model or models at each stage
python3 one_button_active100.py  --experiment experiment-11.15 --nb_stage 4 --nb_round 4 --val_version 16 --nb_epochs 500 --round 0 --gpu 0
# run on terminal 1: train the second model or models at each stage
python3 one_button_active100.py  --experiment experiment-11.15 --nb_stage 4 --nb_round 4 --val_version 16 --nb_epochs 500 --round 1 --gpu 1
# run on terminal 2: train the third model or models at each stage 
python3 one_button_active100.py  --experiment experiment-11.15 --nb_stage 4 --nb_round 4 --val_version 16 --nb_epochs 500 --round 2 --gpu 2
# run on terminal 3: train the fourth model or models at each stage 
python3 one_button_active100.py  --experiment experiment-11.15 --nb_stage 4 --nb_round 4 --val_version 16 --nb_epochs 500 --round 3 --gpu 3

## Running on 2GPUs
# python3 one_button_group_per_GPU.py  --experiment experiment-11.16 --nb_stage 4 --nb_round 4 --val_version 16 --nb_epochs 1 --group 1 --gpu 3

# run on GPU 2
python3 one_button_group_per_GPU.py  --experiment experiment-11.15 --nb_stage 6 --nb_round 4 --val_version 16 --nb_epochs 500 --group 0 --gpu 2 --start_stage 4
# run on GPU 3
python3 one_button_group_per_GPU.py  --experiment experiment-11.15 --nb_stage 6 --nb_round 4 --val_version 16 --nb_epochs 500 --group 1 --gpu 3 --start_stage 4
# python3 one_button_group_per_GPU.py  --experiment experiment-11.15 --nb_stage 4 --nb_round 4 --val_version 16 --nb_epochs 500 --group 0 --gpu 0 --start_stage 0
