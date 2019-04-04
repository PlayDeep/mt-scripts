#!/bin/bash

data=/data/xqzhou/mt-scripts/multi-pro/test
python3 multi-process-highest_score_in_nbest.py \
    --src $data/train.32k.de.shuf.$1 \
    --ref $data/train.32k.en.shuf.$1 \
    --hyp $data/train.32k.de.shuf.$1.nbest5 \
    --nbest 5 \
    --tmp_filename mttmmp$1 \
    --multi_bleu_script '/home/xqzhou/multi-bleu.pl' \
    --njobs 20 \
