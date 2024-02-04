import argparse
import os
import json
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
parser.add_argument("--train_batch_size", type=int, default=4)
parser.add_argument("--eval_batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--output_dir", default="./model_outputs")
args = parser.parse_args()

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
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    output_dir=output_dir,
    logging_dir=f"{output_dir}/logs"
)
model_config = Model_Config(
    backbone_model=args.backbone,
    with_mlp=args.with_mlp
)

if not os.path.exists(training_config.output_dir):
    os.makedirs(training_config.output_dir)
config_json = {"model_config": model_config.to_dict(), "training_config": training_config.to_dict()}
json.dump(config_json, open(f"{training_config.output_dir}/config.json", "w"), indent=4)

##############
## set GPUs ##
##############
configure_gpu_device(config=training_config)


#####################
## package imports ##
#####################
"""
The following imports needs to be done after configuring gpu devices 
so torch will only see specified devices
"""
import inspect
import torch
from sklearn.metrics import classification_report
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
    test_size=training_config.eval_size,
    sampling_rate=feature_extractor.sampling_rate,
    save_test_to_disk=True, 
    test_file_path=f"{training_config.output_dir}/eval_data.pkl"
)
# import pdb; pdb.set_trace()

######################
## initialize model ##
######################
print(f"Initializing model with {model_config.backbone_model} backbone...")
model_config.num_labels = DatasetInfo[args.dataset]["num_labels"]
model = SpeechEmotionClassifier(model_config)
if training_config.freeze_backbone:
    model.freeze_backbone()
model.print_trainable_parameters()

###########
## train ##
###########
training_args = {
    key: val for key, val in training_config.to_dict().items()
    if key in inspect.signature(TrainingArguments.__init__).parameters
}
training_args = TrainingArguments(**training_args)

trainer = SECTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)
trainer.callback_handler.remove_callback(ProgressCallback)
trainer.callback_handler.add_callback(NoLossLoggingInTerminalCallback)

print("Training...")
# import pdb; pdb.set_trace()
trainer.train()

##########
## test ##
##########
print(f"Testing...")
prediction_outputs = trainer.predict(dataset["test"])
preds = torch.argmax(torch.tensor(prediction_outputs.predictions), dim=1).detach().cpu().numpy()
labels = prediction_outputs.label_ids
label_names = _FEAT_DICT["Emotion"]
report = classification_report(labels, preds, target_names=label_names)
print(report)
with open(f"{training_config.output_dir}/test_classification_report.txt", "w") as outfile:
    outfile.write(f"{report}\n")
print("Finished!")