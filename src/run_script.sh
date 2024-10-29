#!/usr/bin/bash

#$ -S /bin/bash 
#$ -pe smp 1
#$ -cwd
#$ -l h_rt=0:20:00
#$ -r y

. ../venv/bin/activate

export OMP_NUM_THREADS=${NSLOTS:-1}
python "$@" --job-idx $SGE_TASK_ID
