#!/bin/bash

ngpu=$1
updatefreq=$2
model_path=/path/to/save_finetune_model


python /path/to/fairseq/fairseq_cli/hydra_train.py \
       --config-dir /path/to/fairseq/examples/data2vec_noisy/config/audio/finetune \
       --config-name chime4 \
       common.user_dir=examples/data2vec_noisy \
       checkpoint.save_dir=${model_path} \
       hydra.run.dir=${model_path} \
       task.data=/path/to/chime_tsv_file \
       model.w2v_path=/path/to/pretrained_model \
       distributed_training.distributed_world_size=${ngpu} \
       optimization.update_freq=[${updatefreq}] \
       task.normalize=True \
       lr_scheduler.phase_ratio=[0.3,0.2,0.5] \
       common.seed=1234 \