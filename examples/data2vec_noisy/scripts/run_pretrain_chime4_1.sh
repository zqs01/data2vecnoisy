#!/bin/bash

ngpu=$1
updatefreq=$2

model_path=/path/to/save_model

python /path/to/fairseq/fairseq_cli/hydra_train.py \
       --config-dir /path/to/fairseq/examples/data2vec_noisy/config/audio/pretraining \
       --config-name base_librispeech \
       common.user_dir=/path/to/fairseq/examples/data2vec_noisy \
       checkpoint.save_dir=${model_path} \
       hydra.run.dir=${model_path} \
       task.data=/path/to/chime_tsv_file \
       distributed_training.distributed_world_size=${ngpu} \
       optimization.update_freq=[${updatefreq}] \
       optimization.max_update=100000  \
       optimization.lr=[0.0001] \
       distributed_training.ddp_backend=no_c10d \
       dataset.max_tokens=3000000 \
       +task.noise_data_path=/path/to/noise_data_manifest \
       +task.noise_snr=\"0,25\" \
       common.log_interval=200 \
       checkpoint.reset_optimizer=True \
       checkpoint.reset_dataloader=True \
       +model.small_scale=0.1 \
       +model.large_scale=0.15 \
       +model.num_negatives=50 \
