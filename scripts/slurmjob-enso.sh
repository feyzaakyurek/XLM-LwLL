#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:volta:2
#SBATCH --output=%j.stdout
#SBATCH --error=%j.stderr
#SBATCH --job-name=lm-enso-8g

##export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

nvidia-smi

LOG_STDOUT="$SLURM_JOB_ID.stdout"
LOG_STDERR="$SLURM_JOB_ID.stderr"

# Start (or restart) experiment
date >> $LOG_STDOUT
which python >> $LOG_STDOUT
echo "---Beginning program ---" >> $LOG_STDOUT
echo "Exp name     : lm-enso-8g" >> $LOG_STDOUT
echo "SLURM Job ID : $SLURM_JOB_ID" >> $LOG_STDOUT
echo "SBATCH script: slurmjob-enso.sh" >> $LOG_STDOUT

srun python train.py \
--master_port 10102 \
--exp_name test_enso_mlm \
--dump_path ./dumped/ \
--data_path ./data/processed/en-so/processed/en-so \
--lgs "en-so" \
--clm_steps "" \
--mlm_steps "en,so" \
--emb_dim 1024 \
--n_layers 6 \
--n_heads 8 \
--dropout 0.1 \
--attention_dropout 0.1 \
--gelu_activation true \
--batch_size 64 \
--bptt 256 \
--optimizer "adam,lr=0.0001" \
--epoch_size 100000 \
--validation_metrics "_valid_mlm_ppl" \
--stopping_criterion "_valid_mlm_ppl,10" &

wait $!