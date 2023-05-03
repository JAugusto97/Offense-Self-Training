import os
import json
import logging
import time
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.special import softmax
from torch.utils.data import DataLoader, RandomSampler, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding, get_scheduler


class BertDataset(Dataset):
    def __init__(self, encodings: BatchEncoding, labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict:
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

class Transformer:
    def __init__(
        self,
        pretrained_bert_name: Optional[str] = "bert-base-cased",
        max_seq_len: Optional[int] = 128,
        attention_dropout: Optional[float] = 0.1,
        classifier_dropout: Optional[float] = 0.1,
        weight_decay: Optional[float] = 1e-2,
        num_train_epochs: Optional[int] = 2,
        batch_size: Optional[int] = 32,
        learning_rate: Optional[float] = 5e-5,
        warmup_ratio: Optional[float] = 0.15,
        augmentation_type: Optional[bool] = None,
        device: Optional[str] = None,
        seed: Optional[int] = 42,
        exp_name: Optional[str] = None
    ) -> None:
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.pretrained_bert_name = pretrained_bert_name
        self.max_seq_len = max_seq_len
        self.attention_dropout = attention_dropout
        self.classifier_dropout = classifier_dropout
        self.weight_decay = weight_decay
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.tokenizer = self.__init_tokenizer()
        self.num_st_iter = 0
        self.model = self.__init_model(self.attention_dropout, self.classifier_dropout)
        self.seed = seed
        self.exp_name = exp_name
        self.augmentation_type = augmentation_type

    def __init_model(
        self, attention_dropout: Optional[float], classifier_dropout: Optional[float]
    ) -> AutoModelForSequenceClassification:
        model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_bert_name)

        # class attributes referring to dropout are not the same for bert and distilbert
        if "distilbert" in self.pretrained_bert_name:
            if attention_dropout is not None:
                model.config.attention_dropout = attention_dropout
            if self.classifier_dropout is not None:
                model.config.seq_classif_dropout = classifier_dropout

        else:
            if attention_dropout is not None:
                model.config.attention_probs_dropout_prob = attention_dropout
            if classifier_dropout is not None:
                model.config.classifier_dropout = classifier_dropout

        return model

    def __init_tokenizer(self) -> AutoTokenizer:
        # bertweet needs normalized inputs
        if self.pretrained_bert_name == "vinai/bertweet-base":
            tokenizer = AutoTokenizer.from_pretrained(self.pretrained_bert_name, normalize=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.pretrained_bert_name)

        return tokenizer

    def __get_dataloader_from_df(self, df: pd.DataFrame) -> DataLoader:
        texts = df.iloc[:, 0].astype("str").to_list()
        targets = df.iloc[:, 1].astype("category").to_list()

        tokenized_train = self.tokenize(texts)
        dataset = BertDataset(tokenized_train, labels=targets)

        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)

        return dataloader

    def __get_optimizer(self, train_dataloader: DataLoader) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        total_steps = len(train_dataloader) * self.num_train_epochs
        num_warmup_steps = int(total_steps * self.warmup_ratio)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
        )

        self.model.to(self.device)
        return optimizer, scheduler

    def __train(
        self,
        train_dataloader: DataLoader,
        is_student=False,
        clip_grad: Optional[bool] = True,
        dev_dataloader: Optional[DataLoader] = None,
        weak_label_dataloader: Optional[DataLoader] = None,
        unl_to_label_batch_ratio: Optional[float] = None,
    ):
        optimizer, scheduler = self.__get_optimizer(train_dataloader)
        progress_bar = tqdm(range(self.num_train_epochs * len(train_dataloader)), desc="Training")

        best_state_dict = self.model.state_dict()
        lowest_val_loss = 1_000_000
        val_loss = 0.0
        for epoch_i in range(self.num_train_epochs):
            # reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_unl_loss, batch_lab_loss, batch_counts, = (
                0,
                0,
                0,
                0,
                0,
            )

            loss_list = []
            unl_loss_list = []
            lab_loss_list = []
            step_list = []
            unl_step_list = []

            # train loop
            loss_fn = torch.nn.CrossEntropyLoss()
            unl_loss, lab_loss = 0, 0
            for step, batch in enumerate(train_dataloader):
                self.model.train()
                batch_counts += 1
                batch_inputs = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad()
                output = self.model(**batch_inputs)
                # if model is student, train with the noised data aswell
                if is_student:
                    unl_logits = []
                    unl_labels = []

                    for _ in range(unl_to_label_batch_ratio):
                        unl_batch = next(iter(weak_label_dataloader))

                        unl_texts = unl_batch["text"]
                        unl_inputs = self.tokenize(unl_texts)
                        unl_inputs["labels"] = unl_batch["labels"].clone().detach()
                        unl_batch_inputs = {k: v.to(self.device) for k, v in unl_inputs.items()}
                        unl_output = self.model(**unl_batch_inputs)

                        unl_logits.append(unl_output.logits.cpu().detach().numpy())
                        unl_labels.append(unl_inputs["labels"].cpu().detach().numpy())

                        del unl_batch_inputs
                        del unl_output

                    # concatenate the unlabeled batch outputs into a single tensor
                    unl_labels = torch.cat([torch.as_tensor(t) for t in unl_labels])
                    unl_logits = torch.cat([torch.as_tensor(t) for t in unl_logits])

                    # combine unlabeled + labeled loss
                    unl_loss = loss_fn(unl_logits, unl_labels)
                    lab_loss = output.loss
                    loss = lab_loss + unl_loss

                    batch_lab_loss += lab_loss.item()
                    batch_unl_loss += unl_loss.item()

                else:
                    loss = output.loss

                batch_loss += loss.item()
                total_loss += loss.item()

                loss.backward()

                # historic data
                loss_list.append(batch_loss / batch_counts)
                step_list.append(step)
                if is_student:
                    unl_loss_list.append(batch_unl_loss / batch_counts)
                    lab_loss_list.append(batch_lab_loss / batch_counts)
                    unl_step_list.append(unl_to_label_batch_ratio * step)

                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                progress_bar.update(1)
                progress_bar.set_description(f'Training Loss={loss.item():.4f} | Validation Loss={val_loss:.4f}')


            # Calculate the average loss over the entire training data
            self.model.eval()
            avg_train_loss = total_loss / len(train_dataloader)
            val_loss, val_accuracy, _, _ = self.score(dev_dataloader)

            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_state_dict = self.model.state_dict()

        self.model.load_state_dict(best_state_dict) # load best model at the end
        return avg_train_loss

    def predict_batch(self, dataloader: DataLoader) -> List[np.array]:
        self.model.eval()
        all_logits = []

        for batch in dataloader:
            batch_inputs = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                output = self.model(**batch_inputs)
                logits = output.logits
            all_logits.append(logits)

        all_logits = torch.cat(all_logits, dim=0)

        probs = torch.nn.functional.softmax(all_logits, dim=1).cpu().numpy()
        labels = np.argmax(probs, axis=1)

        return probs, labels

    def score(
        self, test_dataloader: DataLoader,
    ) -> Tuple[float, float, float]:
        self.model.eval()

        logits, preds, true_labels, val_loss = [], [], [], []
        for batch in test_dataloader:
            batch_labels = batch["labels"]
            batch_inputs = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                batch_output = self.model(**batch_inputs)
                batch_logits = batch_output.logits

            # get accuracy and loss
            batch_probs = torch.nn.functional.softmax(batch_logits, dim=1).cpu().numpy()
            batch_preds = np.argmax(batch_probs, axis=1)

            val_loss.append(batch_output.loss.item())
            preds.extend(batch_preds)
            logits.append(batch_logits)
            true_labels.extend(batch_labels)

        logits = torch.cat(logits, dim=0)
        true_labels = np.array(true_labels)
        preds = np.array(preds)

        clf_report = classification_report(true_labels, preds, zero_division=0)
        f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
        acc = accuracy_score(true_labels, preds)
        val_loss = np.mean(val_loss)
        
        return val_loss, acc, f1, clf_report


    def tokenize(self, texts: List[str]) -> BatchEncoding:
        tokenized = self.tokenizer(
            texts, truncation=True, padding="max_length", max_length=self.max_seq_len, return_tensors="pt"
        )

        return tokenized

    def fit(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        unlabeled_df: pd.DataFrame,
        dev_df: Optional[pd.DataFrame] = None,
      
    ):
        # get dataloaders
        train_dataloader = self.__get_dataloader_from_df(train_df)
        test_dataloader = self.__get_dataloader_from_df(test_df)

        if dev_df is not None:
            dev_dataloader = self.__get_dataloader_from_df(dev_df)
        else:
            dev_dataloader = test_dataloader

        # train teacher model
        logging.info("Training Base Classifier...")
        start = time.time()
        loss = self.__train(
            train_dataloader=train_dataloader,
            dev_dataloader=dev_dataloader,
            is_student=False,
        )

        val_loss, acc, f1, clf_report = self.score(test_dataloader)

        end = time.time()
        logging.info("Classification Report\n" + clf_report)
        logging.info(f"Macro F1-Score: {f1*100:.2f}% - Accuracy: {acc*100:.2f}%")
        logging.info(f"Model {self.num_st_iter} runtime: {(end-start)/60:.2f} minutes.")

        
        text = unlabeled_df.loc[:, "text"].to_list()
        text_tokenized = self.tokenize(text)

        weaklabelset = BertDataset(encodings=text_tokenized, labels=None)
        sampler = RandomSampler(weaklabelset)
        unlabeled_dataloader = DataLoader(weaklabelset, sampler=sampler, batch_size=self.batch_size)
        inferred_labels, inferred_probs = self.predict_batch(unlabeled_dataloader)

        text_augmented = unlabeled_df.loc[:, self.augmentation_type].to_list()
        text_aug_tokenized = self.tokenize(text_augmented)
        weaklabelset_aug = BertDataset(encodings=text_aug_tokenized, labels=None)
        sampler_aug = RandomSampler(weaklabelset_aug)
        unlabeled_dataloader_aug = DataLoader(weaklabelset_aug, sampler=sampler_aug, batch_size=self.batch_size)
        inferred_labels_aug, inferred_probs_aug = self.predict_batch(unlabeled_dataloader_aug)
        
        import ipdb; ipdb.set_trace()
        
        
