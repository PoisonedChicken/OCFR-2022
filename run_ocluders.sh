#!/bin/sh
#​
#SBATCH -p gpu_min32GB   # Global GPU partition (i.e., any GPU)​
#SBATCH --qos gpu_min32GB       # QoS level
#SBATCH -t 2-00:00               # Time(D-HH:MM)​
#SBATCH --job-name ocluders   # Job name​
#SBATCH -o slurm.%N.%j.out      # STDOUT​
#SBATCH -e slurm.%N.%j.err      # STDERR​

python /nas-ctm01/homes/rcmaia/OCFR/OCFR-2022/align_db.py --input-dir /nas-ctm01/datasets/public/BIOMETRICS/Face_Recognition/synthetic_data/ca-cpd25-synthetic-uniform-10050_aligned/ --output-dir /nas-ctm01/datasets/public/BIOMETRICS/Face_Recognition/synthetic_data/occluded_ca-cpd25-synthetic-uniform-10050/Protocol4/

python /nas-ctm01/homes/rcmaia/python_notifications.py --message 'Finished OCCLUDERS! - Synth protocol4'