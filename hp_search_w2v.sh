#!/bin/bash

echo "w2v-bert_frozen"
python hp_search.py --devices 0 --freeze_backbone

echo "w2v-bert_frozen_mlp"
python hp_search.py --devices 0 --freeze_backbone --with_mlp

echo "w2v_frozen"
python hp_search.py --devices 0 --backbone facebook/wav2vec2-base-960h --freeze_backbone

echo "w2v_frozen_mlp"
python hp_search.py --devices 0 --backbone facebook/wav2vec2-base-960h --freeze_backbone --with_mlp

echo "finetune w2v-bert"
python hp_search.py --devices 0

echo "finetune w2v"
python hp_search.py --devices 0 --backbone facebook/wav2vec2-base-960h