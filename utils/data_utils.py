# TODO: add proper logging

import os
from typing import Dict, List, Type, Optional
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset
import torch
from transformers import BatchEncoding


class BertDataset(Dataset):
    def __init__(self, encodings: Type[BatchEncoding], labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


class InferredDataset(Dataset):
    def __init__(self, text: str, augmented_text: Optional[str], labels=None):
        self.text = text
        self.augmented_text = augmented_text
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict:
        item = {"text": self.text[idx]}

        if self.labels is not None:
            item["labels"] = self.labels[idx]
        if self.augmented_text is not None:
            item["augmented_text"] = self.augmented_text[idx]

        return item

    def __len__(self) -> int:
        return len(self.text)


def load_mhs(data_path):
    path = os.path.join(data_path, "english", "measuring_hate_speech")

    unlabeled_path = os.path.join(data_path, "english", "unlabeled", "tweets_augmented.csv")
    train_path = os.path.join(path, "measuring_hate_speech.csv")

    data_df = pd.read_csv(train_path)
    data_df.loc[data_df["hate_speech_score"] >= 1, "label"] = 1
    data_df.loc[data_df["hate_speech_score"] < 1, "label"] = 0
    data_df = data_df[["text", "label"]]
    data_df["label"] = data_df["label"].astype(int)

    train_df, dev_df = train_test_split(data_df, train_size=0.7, stratify=data_df["label"], random_state=CFG.seed)
    dev_df, test_df = train_test_split(dev_df, train_size=0.5, stratify=dev_df["label"], random_state=CFG.seed)

    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    unlabeled_df = pd.read_csv(unlabeled_path)

    return train_df, dev_df, test_df, unlabeled_df


def load_convabuse(data_path):
    path = os.path.join(data_path, "english", "ConvAbuse")

    unlabeled_path = os.path.join(data_path, "english", "unlabeled", "tweets_augmented.csv")
    train_path = os.path.join(path, "ConvAbuseEMNLPtrain.csv")
    dev_path = os.path.join(path, "ConvAbuseEMNLPvalid.csv")
    test_path = os.path.join(path, "ConvAbuseEMNLPtest.csv")

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)
    unlabeled_df = pd.read_csv(unlabeled_path)

    train_df["text"] = train_df.apply(
        lambda x: x["prev_agent"] + "\n" + x["prev_user"] + "\n" + x["agent"] + "\n" + x["user"],
        axis=1,
    )
    dev_df["text"] = dev_df.apply(
        lambda x: x["prev_agent"] + "\n" + x["prev_user"] + "\n" + x["agent"] + "\n" + x["user"],
        axis=1,
    )
    test_df["text"] = test_df.apply(
        lambda x: x["prev_agent"] + "\n" + x["prev_user"] + "\n" + x["agent"] + "\n" + x["user"],
        axis=1,
    )

    train_df = train_df[["text", "is_abuse_majority"]]
    dev_df = dev_df[["text", "is_abuse_majority"]]
    test_df = test_df[["text", "is_abuse_majority"]]

    return train_df, dev_df, test_df, unlabeled_df


def load_olid(data_path):
    eng_path = os.path.join(data_path, "english")

    train_path = os.path.join(eng_path, "OLIDv1.0", "olid-training-v1.0.tsv")
    test_path = os.path.join(eng_path, "OLIDv1.0", "testset-levela.tsv")
    test_labels_path = os.path.join(eng_path, "OLIDv1.0", "labels-levela.csv")
    unlabeled_path = os.path.join(eng_path, "unlabeled", "tweets_augmented.csv")

    train_df = pd.read_csv(train_path, engine="python", sep="\t")[["tweet", "subtask_a"]]
    train_df["subtask_a"] = train_df["subtask_a"].apply(lambda x: 1 if x == "OFF" else 0)
    train_df = train_df.rename({"tweet": "text", "subtask_a": "toxic"}, axis=1)

    test_df = pd.read_csv(test_path, engine="python", sep="\t")
    test_labels = pd.read_csv(test_labels_path, header=None)
    test_df["toxic"] = test_labels[1].apply(lambda x: 1 if x == "OFF" else 0)
    test_df = test_df[["tweet", "toxic"]]
    test_df = test_df.rename({"tweet": "text"}, axis=1)

    unlabeled_df = pd.read_csv(unlabeled_path)[["text", "text_augmented"]]
    unlabeled_df["text"] = unlabeled_df["text"]
    unlabeled_df["text_augmented"] = unlabeled_df["text_augmented"]

    unlabeled_df = unlabeled_df.drop_duplicates("text")

    return train_df, None, test_df, unlabeled_df


def load_davidson(data_path):
    path = os.path.join(data_path, "english", "davidson", "davidson.csv")
    unlabeled_path = os.path.join(data_path, "english", "unlabeled", "tweets_augmented.csv")

    data_df = pd.read_csv(path)
    data_df.loc[data_df["class"] != 0, "label"] = 1
    data_df.loc[data_df["class"] == 0, "label"] = 0
    data_df["label"] = data_df["label"].astype(int)
    data_df = data_df[["tweet", "label"]]

    train_df, dev_df = train_test_split(data_df, train_size=0.7, stratify=data_df["label"], random_state=CFG.seed)
    dev_df, test_df = train_test_split(dev_df, train_size=0.5, stratify=dev_df["label"], random_state=CFG.seed)
    unlabeled_df = pd.read_csv(unlabeled_path)

    train_df = train_df.reset_index(drop=True)
    aux_train_df = train_df[train_df["label"] == 0]
    train_df = aux_train_df.append(train_df[train_df["label"] == 1].sample(len(aux_train_df)))
    dev_df = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, dev_df, test_df, unlabeled_df


def get_stratified_split(df, num_split, seed):
    """splits the dataset into 4 equal sized stratified parts and returns one of them"""
    splits = []
    left_half, right_half = train_test_split(
        df, train_size=0.5, shuffle=True, stratify=df.iloc[:, 1], random_state=seed
    )
    splits.extend(
        train_test_split(
            left_half,
            train_size=0.5,
            shuffle=True,
            stratify=left_half.iloc[:, 1],
            random_state=seed,
        )
    )
    splits.extend(
        train_test_split(
            right_half,
            train_size=0.5,
            shuffle=True,
            stratify=right_half.iloc[:, 1],
            random_state=seed,
        )
    )

    return splits[num_split]


def load_dataset(dataset_name, few_shot=False, num_split=None):
    if dataset_name == "olidv1":
        train_df, dev_df, test_df, unlabeled_df = load_olid()
    elif dataset_name == "convabuse":
        train_df, dev_df, test_df, unlabeled_df = load_convabuse()
    elif dataset_name == "davidson":
        train_df, dev_df, test_df, unlabeled_df = load_davidson()
    elif dataset_name == "measuring_hate_speech":
        train_df, dev_df, test_df, unlabeled_df = load_mhs()

    if few_shot:
        train_df = get_stratified_split(train_df, num_split)

    loaded_log = f"""\tLoaded {dataset_name}"""

    if few_shot:
        loaded_log += f" - Split {num_split}"

    loaded_log += f"""\n
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
    log(loaded_log)

    return train_df, dev_df, test_df, unlabeled_df
