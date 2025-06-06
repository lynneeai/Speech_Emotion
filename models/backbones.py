import torch.nn as nn
from transformers import WhisperModel, Wav2Vec2BertModel, Wav2Vec2Model
from configs import Model_Config


class WhisperEncoderBackbone(nn.Module):
    def __init__(self, config: Model_Config):
        super(WhisperEncoderBackbone, self).__init__()

        self.model = WhisperModel.from_pretrained(config.backbone_model).get_encoder()
    
    def forward(self, input_features, attention_mask=None):
        outputs = self.model(input_features, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def freeze(self):
        self.model._freeze_parameters()

    def get_hidden_size(self):
        return self.model.config.hidden_size
    
    
class Wav2Vec2BertBackbone(nn.Module):
    def __init__(self, config: Model_Config):
        super(Wav2Vec2BertBackbone, self).__init__()

        self.model = Wav2Vec2BertModel.from_pretrained(config.backbone_model)

    def forward(self, input_features, attention_mask=None):
        outputs = self.model(input_features, attention_mask=attention_mask)
        return outputs.last_hidden_state
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def get_hidden_size(self):
        return self.model.config.hidden_size
    

class Wav2Vec2Backbone(nn.Module):
    def __init__(self, config: Model_Config):
        super(Wav2Vec2Backbone, self).__init__()

        self.model = Wav2Vec2Model.from_pretrained(config.backbone_model)

    def forward(self, input_features, attention_mask=None):
        outputs = self.model(input_features, attention_mask=attention_mask)
        return outputs.last_hidden_state
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.freeze_feature_encoder()
    
    def get_hidden_size(self):
        return self.model.config.hidden_size