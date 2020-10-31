#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:volta:2
#SBATCH --output=%j.stdout
#SBATCH --error=%j.stderr
#SBATCH --job-name=nmt-kk-8g-newC

##export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

nvidia-smi

LOG_STDOUT="$SLURM_JOB_ID.stdout" 
LOG_STDERR="$SLURM_JOB_ID.stderr"

# Start (or restart) experiment
date >> $LOG_STDOUT
which python >> $LOG_STDOUT
echo "---Beginning program ---" >> $LOG_STDOUT
echo "Exp name     : nmt-kk-8g-newC" >> $LOG_STDOUT
echo "SLURM Job ID : $SLURM_JOB_ID" >> $LOG_STDOUT
echo "SBATCH script: slurmjob-enkk-nmt.sh" >> $LOG_STDOUT

srun python train.py \
--master_port 10102 \
--exp_name unsupMT_kken_10M \
--dump_path ./dumped/ \
--reload_model 'dumped/unsupMT_kken_10M/8057312/best-valid_kk-en_mt_bleu.pth,dumped/unsupMT_kken_10M/8057312/best-valid_kk-en_mt_bleu.pth' \
--data_path ./data/processed/en-kk/en-kk-10M/en-kk \
--lgs "en-kk" \
--mt_steps 'en-kk,kk-en' \
--bt_steps 'en-kk-en,kk-en-kk' \
--encoder_only false \
--emb_dim 1024 \
--n_layers 6 \
--n_heads 8 \
--dropout 0.1 \
--attention_dropout 0.1 \
--gelu_activation true \
--tokens_per_batch 3000 \
--batch_size 32 \
--bptt 256 \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
--epoch_size 100000 \
--eval_bleu true \
--validation_metrics "valid_kk-en_mt_bleu" \
--stopping_criterion "valid_kk-en_mt_bleu,10" &

wait $!

# --reload_model 'dumped/test_enar_mlm/2809757/best-valid_mlm_ppl.pth,dumped/test_enar_mlm/2809757/best-valid_mlm_ppl.pth' \
# --lambda_ae '0:1,100000:0.1,300000:0' \
# --word_shuffle 3 \
# --word_dropout 0.1 \
# --word_blank 0.1 \