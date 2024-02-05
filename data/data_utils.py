import datasets
import pickle
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers import SeamlessM4TFeatureExtractor, WhisperFeatureExtractor, Wav2Vec2FeatureExtractor

from data.ravdess import _FEAT_DICT


DatasetInfo = {
    "ravdess": {
        "data_script_path": "data/ravdess.py",
        "num_labels": 8,
        "label_names": _FEAT_DICT["Emotion"]
    },
    "iemocap": {
        "data_script_path": "minoosh/IEMOCAP_Speech_dataset",
        "num_labels": 4,
        "label_names": ["angry", "happy", "neutral", "sad"]
    }
}


def load_data(data_script_path, test_size, sampling_rate=16000, save_test_to_disk=False, test_file_path=None):
    if "ravdess" in data_script_path:
        dataset = datasets.load_dataset(data_script_path, split="train")
    elif "IEMOCAP" in data_script_path:
        dataset = datasets.load_dataset(
            data_script_path, 
            split="Session1+Session2+Session3+Session4+Session5"
        )
        dataset = dataset.rename_column("emotion", "labels")
    else:
        raise Exception("Unsupported dataset.")
        
    dataset = dataset.train_test_split(test_size=test_size)
    if save_test_to_disk:
        pickle.dump(dataset["test"], open(test_file_path, "wb"))
    dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=sampling_rate))
    return dataset
        

def get_feature_extractor(backbone_model):
    if "whisper" in backbone_model:
        feature_extractor = WhisperFeatureExtractor.from_pretrained(backbone_model)
    elif "w2v-bert-2.0" in backbone_model:
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(backbone_model)
    elif "wav2vec2" in backbone_model:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(backbone_model)
    else:
        raise Exception(f"Unsupported backbone: {backbone_model}")

    return feature_extractor


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        feature_extractor (:class:`~transformers.SeamlessM4TFeatureExtractor` or :class:`~transformers.WhisperFeatureExtractor`)
            The feature extractor. Use SeamlessM4TFeatureExtractor for Wav2Vec2Bert and WhisperFeatureExtractor for Whisper
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    feature_extractor: Union[SeamlessM4TFeatureExtractor, WhisperFeatureExtractor, Wav2Vec2FeatureExtractor]
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        sampling_rate = self.feature_extractor.sampling_rate
        if isinstance(self.feature_extractor, Wav2Vec2FeatureExtractor):
            input_features = [
                {
                    "input_values": self.feature_extractor(
                        feature["audio"]["array"], sampling_rate=sampling_rate
                    )["input_values"][0]
                } 
                for feature in features
            ]
        else:
            input_features = [
                {
                    "input_features": self.feature_extractor(
                        feature["audio"]["array"], sampling_rate=sampling_rate
                    )["input_features"][0]
                } 
                for feature in features
            ]

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        if isinstance(self.feature_extractor, Wav2Vec2FeatureExtractor):
            batch["input_features"] = batch["input_values"]
            del batch["input_values"]
        
        labels = torch.LongTensor([feature["labels"] for feature in features])
        batch["labels"] = labels

        return batch