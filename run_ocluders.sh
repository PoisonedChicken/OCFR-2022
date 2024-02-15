#!/bin/sh
#​
#SBATCH -p gpu_min12GB   # Global GPU partition (i.e., any GPU)​
#SBATCH --qos gpu_min12GB       # QoS level
#SBATCH -t 2-00:00               # Time(D-HH:MM)​
#SBATCH --job-name ocluders   # Job name​
#SBATCH -o slurm.%N.%j.out      # STDOUT​
#SBATCH -e slurm.%N.%j.err      # STDERR​

python /nas-ctm01/homes/rcmaia/OCFR/OCFR-2022/align_db.py --input-dir /nas-ctm01/datasets/public/BIOMETRICS/Face_Recognition/rfw/test_aligned/data/African/ --output-dir /nas-ctm01/datasets/public/BIOMETRICS/Face_Recognition/rfw_ocfr/African/

python /nas-ctm01/homes/rcmaia/python_notifications.py --message 'Finished OCCLUDERS! - African'