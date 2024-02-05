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


## Experiments
### RAVDESS
|                                    |    Acc   |    F1    |
|------------------------------------|:--------:|:--------:|
| Fine-tuned Wav2Vec2+ClassHead      |   0.64   |   0.57   |
| Frozen Wav2Vec2+ClassHead          |   0.26   |          |
| Frozen Wav2Vec2+MLP+ClassHead      |   0.20   |   0.11   |
| Fine-tuned Wav2Vec2-Bert+ClassHead |   0.19   |   0.11   |
| Frozen Wav2Vec2-Bert+ClassHead     |   0.46   |   0.42   |
| Frozen Wav2Vec2-Bert+MLP+ClassHead |   0.44   |   0.38   |
| **Fine-tuned Whisper+ClassHead**   | **0.88** | **0.88** |
| Frozen Whisper+ClassHead           |   0.74   |   0.74   |
| Frozen Whisper+MLP+ClassHead       |   0.81   |   0.81   |

### IEMOCAP
|                                    |    Acc   |    F1    |
|------------------------------------|:--------:|:--------:|
| **Fine-tuned Whisper+ClassHead**   | **0.74** | **0.75** |