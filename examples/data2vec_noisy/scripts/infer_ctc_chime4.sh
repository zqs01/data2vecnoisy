#!/bin/bash
subset=test
path=/path/to/finetuned_model
model=checkpoint_best.pt

mkdir -p ${path}/${subset}_${model}

python /path/to/fairseq/examples/speech_recognition/new/infer.py \
       --config-dir /path/to/fairseq/examples/speech_recognition/new/conf \
       --config-name infer task=audio_finetuning \
       common.user_dir=examples/data2vec_noisy \
       task.data=/path/to/chime_tsv_file \
       task.labels=ltr \
       decoding.type=viterbi \
       dataset.gen_subset=${subset} \
       common_eval.path=${path}/${model} \
       distributed_training.distributed_world_size=1 \
       common_eval.results_path=${path}/${subset}_${model} \
       decoding.results_path=${path}/${subset}_${model} \


# sclite -r ref.word -h hypo.word -i rm -o all stdout > wer.log