import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import EvalPrediction
from transformers.file_utils import ModelOutput
from torcheval.metrics import MulticlassAccuracy

from configs import Model_Config
from .backbones import WhisperEncoderBackbone, Wav2Vec2BertBackbone, Wav2Vec2Backbone


@dataclass
class SpeechClassifierOutput(ModelOutput):
    logits: torch.FloatTensor = None
    loss: torch.FloatTensor = None
        
class ClassificationHeadWithMLP(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout=0.0):
        super(ClassificationHeadWithMLP, self).__init__()
        
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, inputs):
        x = inputs
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class SpeechEmotionClassifier(nn.Module):
    def __init__(self, config: Model_Config):
        super(SpeechEmotionClassifier, self).__init__()
        self.config = config

        if "whisper" in config.backbone_model:
            self.backbone = WhisperEncoderBackbone(config)
        elif "w2v-bert-2.0" in config.backbone_model:
            self.backbone = Wav2Vec2BertBackbone(config)
        elif "wav2vec2" in config.backbone_model:
            self.backbone = Wav2Vec2Backbone(config)
        else:
            raise Exception(f"Unsupported backbone: {config.backbone_model}")
        
        if config.with_mlp:
            self.classifier = ClassificationHeadWithMLP(self.backbone.get_hidden_size(), config.num_labels)
        else:
            self.projector = nn.Linear(self.backbone.get_hidden_size(), config.classifier_proj_size)
            self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        self.loss_fct = nn.CrossEntropyLoss()

    def freeze_backbone(self):
        self.backbone.freeze()
        
    def print_trainable_parameters(self):
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable paramebers: {num_params}")
        
    def pool_outputs(self, x, mode="mean"):
        if mode == "mean":
            return torch.mean(x, dim=1)
        elif mode == "sum":
            return torch.sum(x, dim=1)
        elif mode == "max":
            return torch.max(x, dim=1)[0]
        else:
            raise Exception(f"Unsupported pooling mode: {mode}")

    def forward(self, input_features, labels, attention_mask=None):
        backbone_outputs = self.backbone(input_features, attention_mask=attention_mask)
        if self.config.with_mlp:
            pooled_outputs = self.pool_outputs(backbone_outputs, self.config.pooling_mode)
        else:
            proj_outputs = self.projector(backbone_outputs)
            pooled_outputs = self.pool_outputs(proj_outputs, self.config.pooling_mode)
        
        logits = self.classifier(pooled_outputs)
        loss = self.loss_fct(logits.view(-1, self.config.num_labels), labels)

        return SpeechClassifierOutput(logits=logits, loss=loss)
    
    
def compute_metrics(p: EvalPrediction):
    metric = MulticlassAccuracy()
    
    logits = torch.tensor(p.predictions)
    labels = torch.tensor(p.label_ids)
    metric.update(logits, labels)
    
    return {"accuracy": metric.compute()}