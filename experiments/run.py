import sys

sys.path.append("..")

import os
from noisystudent import NoisyStudent
import argparse
import gdown
from experiments import load_dataset, set_seed, get_logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--seed", default=42)
    parser.add_argument("--exp_name", default="experiment")

    # bert args
    parser.add_argument("--pretrained_bert_name", default="bert-base-cased")
    parser.add_argument("--max_seq_len", default=128)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight_decay", default=1e-2)
    parser.add_argument("--num_train_epochs", default=2)
    parser.add_argument("--learning_rate", default=5e-5)
    parser.add_argument("--warmup_ratio", default=0.15)
    parser.add_argument("--classifier_dropout", default=None)
    parser.add_argument("--attention_dropout", default=None)

    # ST args
    parser.add_argument("--min_confidence_threshold", default=0.51)
    parser.add_argument("--num_st_iters", default=2)
    parser.add_argument("--use_augmentation", default=True)
    parser.add_argument("--increase_attention_dropout_amount", default=None)
    parser.add_argument("--increase_classifier_dropout_amount", default=None)
    parser.add_argument("--increase_confidence_threshold_amount", default=None)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    log_path = os.path.join("logs", f"{args.exp_name}.log")
    logger = get_logger(level="info", filename=log_path)

    set_seed(args.seed)

    logger.debug("Downloading Datasets.")
    folder_url = "https://drive.google.com/drive/u/3/folders/1rGblqA0Wh0vhDFrjasMGvOYpyDiD3jVW"
    gdown.download_folder(id=folder_url, output="./", quiet=True)

    train_df, dev_df, test_df, weak_label_df = load_dataset(args.dataset)

    ns = NoisyStudent(
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
    )

    ns.fit(
        train_df=train_df,
        dev_df=dev_df,
        test_df=test_df,
        unlabeled_df=weak_label_df,
        num_iters=args.num_st_iters,
        min_confidence_threshold=args.min_confidence_threshold,
        increase_attention_dropout_amount=args.increase_attention_dropout_amount,
        increase_classifier_dropout_amount=args.increase_classifier_dropout_amount,
        increase_confidence_threshold_amount=args.increase_confidence_threshold_amount,
        use_augmentation=args.use_augmentation,
    )
