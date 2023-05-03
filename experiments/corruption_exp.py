# %%
import sys

sys.path.append("..")

from selftraining import Transformer

# %%
import os
import argparse
import time
import json
from experiments import load_dataset, set_seed, get_logger
import torch
import wandb

# %%
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("augmentation_type", choices=["backtranslation", "synonym_substitution", "word_swap"])
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--exp_name", default="experiment", type=str)
    parser.add_argument("--loglevel", default="info", type=str)

    # bert args
    parser.add_argument("--pretrained_bert_name", default="bert-base-cased", type=str)
    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--warmup_ratio", default=0.10, type=float)
    parser.add_argument("--classifier_dropout", default=0.1, type=float)
    parser.add_argument("--attention_dropout", default=0.1, type=float)

    args = parser.parse_args()
    return args


# %%
args = get_args()
set_seed(args.seed)

# %%

log_path = os.path.join(
    "logs",
    args.exp_name,
    "default" if args.augmentation_type is None else args.augmentation_type,
    f"seed{args.seed}"
)

train_path = os.path.join(log_path, "train")
test_path = os.path.join(log_path, "test")
if not os.path.exists(log_path):
    os.makedirs(log_path)
if not os.path.exists(train_path):
    os.mkdir(train_path)
if not os.path.exists(test_path):
    os.mkdir(test_path)


logger = get_logger(level=args.loglevel, filename=os.path.join(log_path, "run.log"))
current_device = torch.cuda.current_device()
gpu_name = torch.cuda.get_device_name(current_device) if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {gpu_name}")

train_df, dev_df, test_df, weak_label_df = load_dataset(args.dataset, augmentation_type=args.augmentation_type)
weak_label_df = weak_label_df.sample(500).reset_index(drop=True)

st = Transformer(
    pretrained_bert_name=args.pretrained_bert_name,
    device=args.device,
    classifier_dropout=args.classifier_dropout,
    attention_dropout=args.attention_dropout,
    max_seq_len=args.max_seq_len,
    batch_size=args.batch_size,
    num_train_epochs=args.num_train_epochs,
    learning_rate=args.learning_rate,
    warmup_ratio=args.warmup_ratio,
    weight_decay=args.weight_decay,
    augmentation_type=args.augmentation_type,
    seed=args.seed,
    exp_name=args.exp_name
)

st.fit(
    train_df=train_df,
    dev_df=dev_df,
    test_df=test_df,
    unlabeled_df=weak_label_df
)


