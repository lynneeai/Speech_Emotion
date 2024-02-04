# Speech_Emotion

## Prerequisites
### Python 3.10
```bash
pip install -r requirements.txt
```

To start training
```bash
python train.py --devices 0
```

To tune hyperparameters
```bash
bash hp_search_whisper.sh
```

Currenly only supports single GPU. Still debugging for multi-GPU.