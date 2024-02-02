import datasets
import os
import pandas as pd
from pathlib import Path
from collections import OrderedDict


_FEAT_DICT = OrderedDict([
    ("Modality", ["full-AV", "video-only", "audio-only"]),
    ("Vocal channel", ["speech", "song"]),
    ("Emotion", ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]),
    ("Emotion intensity", ["normal", "strong"]),
    ("Statement", ["Kids are talking by the door", "Dogs are sitting by the door"]),
    ("Repetition", ["1st repetition", "2nd repetition"]),
])
_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"


class RAVDESS(datasets.GeneratorBasedBuilder):
    def _preprocess_ravdess(self, data_path, csv_path):
        def filename2feats(filename):
            codes = filename.stem.split("-")
            d = {}
            for i, k in enumerate(_FEAT_DICT.keys()):
                d[k] = _FEAT_DICT[k][int(codes[i])-1]
            d["Actor"] = codes[-1]
            d["Gender"] = "female" if int(codes[-1]) % 2 == 0 else "male"
            d["Path_to_Wav"] = str(filename)
            return d
        
        data = []
        for actor_dir in Path(data_path).iterdir():
            if actor_dir.is_dir() and "Actor" in actor_dir.name:
                for f in actor_dir.iterdir():
                    data.append(filename2feats(f))

        df = pd.DataFrame(data, columns=list(_FEAT_DICT.keys()) + ["Actor", "Gender", "Path_to_Wav"])
        df.to_csv(csv_path)

    def _info(self):
        return datasets.DatasetInfo(
            description="",
            features=datasets.Features(
                {
                    "audio": datasets.Audio(sampling_rate=48000),
                    "text": datasets.Value("string"),
                    "labels": datasets.ClassLabel(names=_FEAT_DICT["Emotion"]),
                    "emotion_name": datasets.Value("string"),
                    "speaker_id": datasets.Value("string"),
                    "speaker_gender": datasets.Value("string")
                }
            )
        )

    def _split_generators(self, dl_manager):
        archive_path = dl_manager.download_and_extract(_URL)
        csv_path = f"{archive_path}/data.csv"
        self._preprocess_ravdess(archive_path, csv_path)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={"data_info_csv": csv_path}
            ),
        ]

    def _generate_examples(self, data_info_csv):
        data_info = pd.read_csv(open(data_info_csv, encoding="utf8"))
        for audio_idx in range(data_info.shape[0]):
            audio_data = data_info.iloc[audio_idx]
            example = {
                "audio": audio_data["Path_to_Wav"],
                "text": audio_data["Statement"],
                "labels": audio_data["Emotion"],
                "emotion_name": audio_data["Emotion"],
                "speaker_id": audio_data["Actor"],
                "speaker_gender": audio_data["Gender"]
            }

            yield audio_idx, example