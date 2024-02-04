#!/bin/bash

echo "whisper_frozen"
python hp_search.py --devices 1 --backbone openai/whisper-medium --freeze_backbone

echo "whisper_frozen_mlp"
python hp_search.py --devices 1 --backbone openai/whisper-medium --freeze_backbone --with_mlp

echo "finetune whisper"
python hp_search.py --devices 1 --backbone openai/whisper-medium