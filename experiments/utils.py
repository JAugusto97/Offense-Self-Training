import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import random
from typing import Optional, Tuple
import re
import string
import unicodedata

import numpy as np
import torch
import sys


def get_logger(level: Optional[str] = "debug", filename: Optional[str] = None) -> logging.Logger:
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "fatal": logging.FATAL,
    }
    loglevel = level_map.get(level)
    if loglevel is None:
        raise TypeError

    logging.basicConfig(
        level=loglevel,
        filename=filename,
        filemode="w" if filename else None,
        format="%(levelname)s | %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def set_seed(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def load_mhs(seed: Optional[int] = None) -> Tuple[pd.DataFrame]:
    path = os.path.join("datasets", "measuring_hate_speech")
    train_path = os.path.join(path, "measuring_hate_speech.csv")

    data_df = pd.read_csv(train_path)
    data_df.loc[data_df["hate_speech_score"] >= 1, "label"] = 1
    data_df.loc[data_df["hate_speech_score"] < 1, "label"] = 0
    data_df = data_df[["text", "label"]]
    data_df["label"] = data_df["label"].astype(int)

    train_df, dev_df = train_test_split(data_df, train_size=0.7, stratify=data_df["label"], random_state=seed)
    dev_df, test_df = train_test_split(dev_df, train_size=0.5, stratify=dev_df["label"], random_state=seed)

    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, dev_df, test_df


def load_convabuse() -> Tuple[pd.DataFrame]:
    path = os.path.join("datasets", "ConvAbuse")
    train_path = os.path.join(path, "ConvAbuseEMNLPtrain.csv")
    dev_path = os.path.join(path, "ConvAbuseEMNLPvalid.csv")
    test_path = os.path.join(path, "ConvAbuseEMNLPtest.csv")

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    train_df["text"] = train_df.apply(
        lambda x: x["prev_agent"] + "\n" + x["prev_user"] + "\n" + x["agent"] + "\n" + x["user"], axis=1
    )
    dev_df["text"] = dev_df.apply(
        lambda x: x["prev_agent"] + "\n" + x["prev_user"] + "\n" + x["agent"] + "\n" + x["user"], axis=1
    )
    test_df["text"] = test_df.apply(
        lambda x: x["prev_agent"] + "\n" + x["prev_user"] + "\n" + x["agent"] + "\n" + x["user"], axis=1
    )


    train_df = train_df[["text", "is_abuse_majority"]].rename({"is_abuse_majority": "toxic"}, axis=1)
    dev_df = dev_df[["text", "is_abuse_majority"]].rename({"is_abuse_majority": "toxic"}, axis=1)
    test_df = test_df[["text", "is_abuse_majority"]].rename({"is_abuse_majority": "toxic"}, axis=1)
    
    return train_df, dev_df, test_df


def load_olid() -> Tuple[pd.DataFrame]:
    path = os.path.join("datasets", "OLIDv1.0")

    train_path = os.path.join(path, "olid-training-v1.0.tsv")
    test_path = os.path.join(path, "testset-levela.tsv")
    test_labels_path = os.path.join(path, "labels-levela.csv")

    train_df = pd.read_csv(train_path, engine="python", sep="\t")[["tweet", "subtask_a"]]
    train_df["subtask_a"] = train_df["subtask_a"].apply(lambda x: 1 if x == "OFF" else 0)
    train_df = train_df.rename({"tweet": "text", "subtask_a": "toxic"}, axis=1)

    test_df = pd.read_csv(test_path, engine="python", sep="\t")
    test_labels = pd.read_csv(test_labels_path, header=None)
    test_df["toxic"] = test_labels[1].apply(lambda x: 1 if x == "OFF" else 0)
    test_df = test_df[["tweet", "toxic"]]
    test_df = test_df.rename({"tweet": "text"}, axis=1)

    return train_df, None, test_df

def load_waseem() -> Tuple[pd.DataFrame]:
    path = os.path.join("datasets", "waseem")

    train_path = os.path.join(path, "waseem_train.csv")
    dev_path = os.path.join(path, "waseem_dev.csv")
    test_path = os.path.join(path, "waseem_test.csv")

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    return train_df, dev_df, test_df


def load_dataset(dataset_name, augmentation_type):
    if dataset_name == "olidv1":
        train_df, dev_df, test_df = load_olid()
    elif dataset_name == "convabuse":
        train_df, dev_df, test_df = load_convabuse()
    elif dataset_name == "mhs":
        train_df, dev_df, test_df = load_mhs()
    elif dataset_name == "waseem":
        train_df, dev_df, test_df = load_waseem()
    else:
        raise Exception(f"{dataset_name} is not a valid dataset.")

    
    unlabeled_df = pd.read_csv("datasets/tweets_augmented.csv")
    cols = ["text"]
    if augmentation_type is not None:
        cols += [augmentation_type]

    unlabeled_df = unlabeled_df[cols]
    unlabeled_df = unlabeled_df.drop_duplicates(subset=cols)
    unlabeled_df = unlabeled_df.dropna(subset=cols)
    unlabeled_df = unlabeled_df.reset_index(drop=True)

    loaded_log = f"""
            Loaded {dataset_name}
        Train Size: {len(train_df)}
            Positives: {len(train_df[train_df.iloc[:, 1] == 1])}
            Negatives: {len(train_df[train_df.iloc[:, 1] == 0])}
        """

    if dev_df is not None:
        loaded_log += f"""
        Dev Size: {len(dev_df)}
            Positives: {len(dev_df[dev_df.iloc[:, 1] == 1])}
            Negatives: {len(dev_df[dev_df.iloc[:, 1] == 0])}
        """

    loaded_log += f"""
        Test Size: {len(test_df)}
            Positives: {len(test_df[test_df.iloc[:, 1] == 1])}
            Negatives: {len(test_df[test_df.iloc[:, 1] == 0])}
        Augmented Data: {len(unlabeled_df)}
        """
    logging.info(loaded_log)

    return train_df, dev_df, test_df, unlabeled_df

def normalize_tweet(text):
    # remove mentions and URLs
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # remove accents
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # remove extra whitespace
    text = re.sub('\s+', ' ', text).strip()
    return text