import torch.nn as nn
from transformers import WhisperModel, Wav2Vec2BertModel
from configs import Model_Config


class Whisper_Encoder_Backbone(nn.Module):
    def __init__(self, config: Model_Config):
        super(Whisper_Encoder_Backbone, self).__init__()

        self.model = WhisperModel.from_pretrained(config.backbone_model).get_encoder()
    
    def forward(self, input_features, attention_mask=None):
        outputs = self.model(input_features, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def freeze(self):
        self.model._freeze_parameters()

    def get_hidden_size(self):
        return self.model.config.hidden_size
    
    
class Wav2Vec2Bert_Backbone(nn.Module):
    def __init__(self, config: Model_Config):
        super(Wav2Vec2Bert_Backbone, self).__init__()

        self.model = Wav2Vec2BertModel.from_pretrained(config.backbone_model)

    def forward(self, input_features, attention_mask=None):
        outputs = self.model(input_features, attention_mask=attention_mask)
        return outputs.last_hidden_state
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def get_hidden_size(self):
        return self.model.config.hidden_size