import json
from dataclasses import dataclass, asdict


@dataclass
class Model_Config:
    backbone_model: str
    classifier_proj_size: str = 256
    num_labels: str = 8
    pooling_mode: str = "mean"
    with_mlp: bool = False
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items()}
    
    def to_json_string(self):
        return json.dumps(self.to_dict())