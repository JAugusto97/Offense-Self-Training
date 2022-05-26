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


class WeakLabelDataset(Dataset):
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
