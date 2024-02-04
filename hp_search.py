import argparse
from datetime import datetime

from configs import Training_Config, Model_Config
from utils import configure_gpu_device


parser = argparse.ArgumentParser()
# required arguments
parser.add_argument("--devices", required=True, type=str, help="GPU ids separated by comma, e.g. 0,1 or 1")
# optional arguments
parser.add_argument(
    "--backbone", 
    choices=["facebook/w2v-bert-2.0", "openai/whisper-medium", "facebook/wav2vec2-base-960h"], 
    default="facebook/w2v-bert-2.0"
)
parser.add_argument(
    "--dataset",
    choices=["ravdess", "iemocap"],
    default="ravdess"
)
parser.add_argument("--freeze_backbone", action="store_true")
parser.add_argument("--with_mlp", action="store_true")
parser.add_argument("--eval_batch_size", type=int, default=4)
parser.add_argument("--eval_steps", type=int, default=500)
parser.add_argument("--save_steps", type=int, default=500)
parser.add_argument("--output_dir", default="./hp_search_outputs")
args = parser.parse_args()

##############
## set GPUs ##
##############
configure_gpu_device(devices=args.devices)

##########################
## set and save configs ##
##########################
now_dt = datetime.now()
timestamp = now_dt.strftime("%m-%d-%y-%H:%M")

output_dir = f"{args.output_dir}/{args.dataset}/{args.backbone.split('/')[-1]}"
if args.freeze_backbone:
    output_dir += "_frozen"
if args.with_mlp:
    output_dir += "_mlp"
output_dir += f"_{timestamp}"

training_config = Training_Config(
    devices=[int(d) for d in args.devices.split(",")],
    freeze_backbone=args.freeze_backbone,
    per_device_eval_batch_size=args.eval_batch_size,
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
    output_dir=output_dir
)
model_config = Model_Config(
    backbone_model=args.backbone,
    with_mlp=args.with_mlp
)
    

#####################
## package imports ##
#####################
"""
The following imports needs to be done after configuring gpu devices 
so torch will only see specified devices
"""
import copy
import inspect
import json
import glob
import torch
import optuna
from sklearn.metrics import classification_report
from typing import Dict
from transformers import TrainingArguments, ProgressCallback

from trainer import SECTrainer, NoLossLoggingInTerminalCallback
from data.data_utils import DatasetInfo, load_data, get_feature_extractor, DataCollatorWithPadding
from models.classifier import SpeechEmotionClassifier, compute_metrics
from data.ravdess import _FEAT_DICT


##################
## prepare data ##
##################
print("Loading data...")
feature_extractor = get_feature_extractor(model_config.backbone_model)
data_collator = DataCollatorWithPadding(feature_extractor)
dataset = load_data(
    data_script_path=DatasetInfo[args.dataset]["data_script_path"], 
    test_size=0.2,
    sampling_rate=feature_extractor.sampling_rate
)
# import pdb; pdb.set_trace()

###################
## training args ##
###################
training_args = {
    key: val for key, val in training_config.to_dict().items()
    if key in inspect.signature(TrainingArguments.__init__).parameters
}
training_args = TrainingArguments(**training_args)

###############
## hp search ##
###############
def model_init(trial: optuna.Trial):
    print(f"Initializing model with {model_config.backbone_model} backbone...")
    model_config.num_labels = DatasetInfo[args.dataset]["num_labels"]
    model = SpeechEmotionClassifier(model_config) 
    if training_config.freeze_backbone:
        model.freeze_backbone()
    model.print_trainable_parameters()
    return model

def optuna_hp_space(trial: optuna.Trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 10, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16])
    }
    
def compute_objective(metrics: Dict[str, float]):
    metrics = copy.deepcopy(metrics)
    acc = metrics.pop("eval_accuracy", None)
    return acc

trainer = SECTrainer(
    model=None,
    model_init=model_init,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)
trainer.callback_handler.remove_callback(ProgressCallback)
trainer.callback_handler.add_callback(NoLossLoggingInTerminalCallback)

best_trials = trainer.hyperparameter_search(
    direction="maximize",
    compute_objective=compute_objective,
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=20,
)

#################################
## save results for best trial ##
#################################
json.dump(
    {**{"run_id": best_trials.run_id}, **best_trials.hyperparameters}, 
    open(f"{output_dir}/best_run_hp.json", "w"), 
    indent=4
)

ckpt_dir = glob.glob(f"{output_dir}/run-{best_trials.run_id}/checkpoint*")[0]
trainer._load_from_checkpoint(ckpt_dir)
prediction_outputs = trainer.predict(dataset["test"])
preds = torch.argmax(torch.tensor(prediction_outputs.predictions), dim=1).detach().cpu().numpy()
labels = prediction_outputs.label_ids
label_names = _FEAT_DICT["Emotion"]
report = classification_report(labels, preds, target_names=label_names)
print(report)
with open(f"{output_dir}/best_run_classification_report.txt", "w") as outfile:
    outfile.write(f"{report}\n")
print("Finished!")