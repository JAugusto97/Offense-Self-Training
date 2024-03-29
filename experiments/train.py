import sys

sys.path.append("..")

import os
from selftraining import SelfTrainer
import argparse
import time
import json
from experiments import load_dataset, set_seed, get_logger
import torch
import wandb


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--augmentation_type", choices=["backtranslation", "synonym_substitution", "word_swap"], default=None, nargs="?", const=None)
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

    # ST args
    parser.add_argument("--min_confidence_threshold", default=0.8, type=float)
    parser.add_argument("--num_st_iters", default=3, type=int)
    parser.add_argument("--increase_attention_dropout_amount", default=None, type=float)
    parser.add_argument("--increase_classifier_dropout_amount", default=None, type=float)
    parser.add_argument("--increase_confidence_threshold_amount", default=None, type=float)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

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

    wandb.init(
        project="selftrain",
        name=args.dataset + f"-{'default' if args.augmentation_type is None else args.augmentation_type}",
        config=vars(args)
)

    logger = get_logger(level=args.loglevel, filename=os.path.join(log_path, "run.log"))
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device) if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {gpu_name}")
    
    train_df, dev_df, test_df, weak_label_df = load_dataset(args.dataset, augmentation_type=args.augmentation_type)

    st = SelfTrainer(
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

    start = time.time()
    base_f1, best_f1, best_iter = st.fit(
        train_df=train_df,
        dev_df=dev_df,
        test_df=test_df,
        unlabeled_df=weak_label_df,
        num_iters=args.num_st_iters,
        min_confidence_threshold=args.min_confidence_threshold,
        increase_attention_dropout_amount=args.increase_attention_dropout_amount,
        increase_classifier_dropout_amount=args.increase_classifier_dropout_amount,
        increase_confidence_threshold_amount=args.increase_confidence_threshold_amount
    )
    end = time.time()
    runtime = end - start

    improvement_f1 = best_f1 - base_f1
    wandb.log({"best_f1": best_f1, "base_f1": base_f1, "improvement_f1": improvement_f1})

    logger.info(f"Base Model F1-Macro: {100*base_f1:.2f}%")
    logger.info(f"Best Model F1-Macro: {100*best_f1:.2f}%")
    logger.info(f"Improvement F1-Macro: {100*improvement_f1:.2f}%")
    logger.info(f"\nTotal Runtime: {runtime/60:.2f} minutes.")

    d = {"base_f1": str(base_f1), "best_f1": str(best_f1), "best_iter": str(best_iter)}
    with open(
        f"logs/results-" +
            f"{args.exp_name}-" +
            ("default" if args.augmentation_type is None else args.augmentation_type) +
            f"-seed{args.seed}" +
            ".json",
        "w"
    ) as f:
        json.dump(d, f)

    wandb.finish()
