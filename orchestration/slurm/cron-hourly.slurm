#!/bin/bash
#SBATCH --job-name=cron-hourly       # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=0:30:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --partition=PARTITION     # edit me
#SBATCH --account=GROUP@PARTITION # edit me

# do not set -e - we must run all of it
# set -x -e

cd $WORK/cron/scheduler

# ensure to restart self first
sbatch --begin=now+1hour cron-hourly.slurm

# now launch any slurm scripts in cron.hourly
cd $WORK/cron/cron.hourly
for f in *.slurm; do
  sbatch "$f"
done
