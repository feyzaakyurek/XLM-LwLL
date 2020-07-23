# updated number of layers
export NGPU=2

python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
--exp_name unsupMT_aren \
--dump_path ./dumped/ \
--reload_model 'dumped/test_enar_mlm/xlmr17/mlm_17_1280.pth,dumped/test_enar_mlm/xlmr17/mlm_17_1280.pth' \
--data_path ./data/data17lang/processed/ar-en \
--lgs 'ar-en' \
--ae_steps 'ar,en' \
--bt_steps 'ar-en-ar,en-ar-en' \
--word_shuffle 3 \
--word_dropout 0.1 \
--word_blank 0.1 \
--lambda_ae '0:1,100000:0.1,300000:0' \
--encoder_only false \
--emb_dim 1280 \
--n_layers 6 \
--n_heads 8 \
--dropout 0.1 \
--attention_dropout 0.1 \
--gelu_activation true \
--tokens_per_batch 2000 \
--batch_size 32 \
--bptt 256 \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
--epoch_size 200000 \
--eval_bleu true \
--max_vocab 200000 \
--stopping_criterion 'valid_ar-en_mt_bleu,10' \
--validation_metrics 'valid_ar-en_mt_bleu' # > train.log 2> train.err