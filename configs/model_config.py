import json
from dataclasses import dataclass, asdict


@dataclass
class Model_Config:
    backbone_model: str = "facebook/w2v-bert-2.0" # "openai/whisper-medium"
    classifier_proj_size: str = 256
    num_labels: str = 8
    pooling_mode: str = "mean"
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items()}
    
    def to_json_string(self):
        return json.dumps(self.to_dict())