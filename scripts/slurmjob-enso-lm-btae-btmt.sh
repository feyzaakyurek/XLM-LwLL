#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:volta:2
#SBATCH --output=%j.stdout
#SBATCH --error=%j.stderr
#SBATCH --job-name=4gbtaebtmtso

##export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# nvidia-smi

# LOG_STDOUT="$SLURM_JOB_ID.lm.stdout"
# LOG_STDERR="$SLURM_JOB_ID.lm.stderr"

# # Start (or restart) experiment
# date >> $LOG_STDOUT
# which python >> $LOG_STDOUT
# echo "---Beginning program (LM)---" >> $LOG_STDOUT
# echo "Exp name     : 4glmbtaebtmtso" >> $LOG_STDOUT
# echo "SLURM Job ID : $SLURM_JOB_ID" >> $LOG_STDOUT
# echo "SBATCH script: slurmjob-enso-lm-btae-btmt.sh" >> $LOG_STDOUT

# srun python train.py \
# --master_port 10102 \
# --exp_name test_enso_mlm \
# --dump_path ./dumped/ \
# --data_path ./data/processed/en-so/processed/en-so \
# --lgs "en-so" \
# --clm_steps "" \
# --mlm_steps "en,so" \
# --emb_dim 1024 \
# --n_layers 6 \
# --n_heads 8 \
# --dropout 0.1 \
# --attention_dropout 0.1 \
# --gelu_activation true \
# --batch_size 64 \
# --bptt 256 \
# --optimizer "adam,lr=0.0001" \
# --epoch_size 100000 \
# --validation_metrics "_valid_mlm_ppl" \
# --stopping_criterion "_valid_mlm_ppl,10" &

# wait $!




# # UNSUPERVISED MT bt+ae

# LOG_STDOUT="$SLURM_JOB_ID.stdout" 
# LOG_STDERR="$SLURM_JOB_ID.stderr"
# RELOAD_MODEL="dumped/test_enso_mlm/8176529/best-valid_mlm_ppl.pth,dumped/test_enso_mlm/8176529/best-valid_mlm_ppl.pth"

# # Start (or restart) experiment
# date >> $LOG_STDOUT
# which python >> $LOG_STDOUT
# echo "---Beginning program (BT+AE)---" >> $LOG_STDOUT
# echo "Exp name     : nmt-en-so-4g" >> $LOG_STDOUT
# echo "SLURM Job ID : $SLURM_JOB_ID" >> $LOG_STDOUT
# echo "SBATCH script: slurmjob-enso-lm-btae-btmt.sh" >> $LOG_STDOUT



# srun python train.py \
# --master_port 10102 \
# --exp_name unsupMT_soen \
# --dump_path ./dumped/ \
# --reload_model $RELOAD_MODEL \
# --data_path ./data/processed/en-so/processed/en-so \
# --lgs "en-so" \
# --ae_steps 'en,so' \
# --bt_steps 'en-so-en,so-en-so' \
# --word_shuffle 3 \
# --word_dropout 0.1 \
# --word_blank 0.1 \
# --lambda_ae '0:1,100000:0.1,300000:0' \
# --encoder_only false \
# --emb_dim 1024 \
# --n_layers 6 \
# --n_heads 8 \
# --dropout 0.1 \
# --attention_dropout 0.1 \
# --gelu_activation true \
# --tokens_per_batch 3000 \
# --batch_size 32 \
# --bptt 256 \
# --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
# --epoch_size 100000 \
# --eval_bleu true \
# --validation_metrics "valid_so-en_mt_bleu" \
# --stopping_criterion "valid_so-en_mt_bleu,10" &

# wait $!



# COMPARABLE MT bt+mt

# LOG_STDOUT="$SLURM_JOB_ID.btmt.stdout" 
# LOG_STDERR="$SLURM_JOB_ID.btmt.stderr"
# RELOAD_MODEL="dumped/unsupMT_soen/${SLURM_JOB_ID}/best-valid_so-en_mt_bleu.pth,dumped/unsupMT_soen/${SLURM_JOB_ID}/best-valid_so-en_mt_bleu.pth"

# # Start (or restart) experiment
# date >> $LOG_STDOUT
# which python >> $LOG_STDOUT
# echo "---Beginning program (BT MT)---" >> $LOG_STDOUT
# echo "Exp name     : nmt-en-so-2g" >> $LOG_STDOUT
# echo "SLURM Job ID : $SLURM_JOB_ID" >> $LOG_STDOUT
# echo "SBATCH script: slurmjob-enso-lm-btae-btmt.sh" >> $LOG_STDOUT

# python train.py \
# --master_port 10102 \
# --exp_name unsupMT_soen_comparable \
# --dump_path ./dumped/ \
# --reload_model $RELOAD_MODEL \
# --data_path ./data/processed/en-so/processed/en-so \
# --lgs "en-so" \
# --mt_steps 'en-so,so-en' \
# --bt_steps 'en-so-en,so-en-so' \
# --encoder_only false \
# --emb_dim 1024 \
# --n_layers 6 \
# --n_heads 8 \
# --dropout 0.1 \
# --attention_dropout 0.1 \
# --gelu_activation true \
# --tokens_per_batch 3000 \
# --batch_size 32 \
# --bptt 256 \
# --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
# --epoch_size 100000 \
# --eval_bleu true \
# --validation_metrics "valid_so-en_mt_bleu" \
# --stopping_criterion "valid_so-en_mt_bleu,10" &

# wait $!





# Debugging below.
# RELOAD_MODEL="dumped/test_enso_mlm/8176519/best-valid_mlm_ppl.pth,dumped/test_enso_mlm/8176519/best-valid_mlm_ppl.pth"
# python train.py \
# --master_port 10102 \
# --exp_name unsupMT_soen \
# --dump_path ./dumped/ \
# --reload_model $RELOAD_MODEL \
# --data_path ./data/processed/en-so/processed/en-so \
# --lgs "en-so" \
# --ae_steps 'en,so' \
# --bt_steps 'en-so-en,so-en-so' \
# --word_shuffle 3 \
# --word_dropout 0.1 \
# --word_blank 0.1 \
# --lambda_ae '0:1,100000:0.1,300000:0' \
# --encoder_only false \
# --emb_dim 1024 \
# --n_layers 6 \
# --n_heads 8 \
# --dropout 0.1 \
# --attention_dropout 0.1 \
# --gelu_activation true \
# --tokens_per_batch 3000 \
# --batch_size 32 \
# --bptt 256 \
# --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
# --epoch_size 6000 \
# --eval_bleu true \
# --validation_metrics "valid_so-en_mt_bleu" \
# --stopping_criterion "valid_so-en_mt_bleu,10"