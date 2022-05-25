# TODO:
# - proper logging
# - add noisy loop
# - integrate methods
# - unify __evaluate and score methods

from typing import Dict, List, Optional, Tuple, Type
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, BatchEncoding, get_scheduler
import torch
from torch.utils.data import RandomSampler, DataLoader
import pandas as pd
from data_utils import BertDataset, InferredDataset
from tqdm import tqdm
import time
import json
import numpy as np
from sklearn.metrics import f1_score


class NoisyStudent:
    def __init__(
        self,
        pretrained_bert_name: str = "bert-base-cased",
        max_seq_len: Optional[int] = 256,
        attention_dropout: Optional[float] = None,
        classifier_dropout: Optional[float] = None,
        weight_decay: Optional[float] = 1e-2,
        num_train_epochs: Optional[int] = 2,
        learning_rate: Optional[float] = 5e-5,
        warmup_ratio: Optional[float] = 0.15,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.pretrained_bert_name = pretrained_bert_name
        self.max_seq_len = max_seq_len
        self.attention_dropout = attention_dropout
        self.classifier_dropout = classifier_dropout
        self.weight_decay = weight_decay
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.tokenizer = self.__init_tokenizer()
        self.num_noisy_iteration = 0
        self.model = self.__init_model()

    def __init_model(
        self,
    ) -> torch.AutoModelForSequenceClassification:
        model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_bert_name)

        # class attributes referring to dropout are not the same for bert and distilbert
        if "distilbert" in self.pretrained_bert_name:
            if self.attention_dropout is not None:
                model.config.attention_dropout = self.attention_dropout
            if self.classifier_dropout is not None:
                model.config.seq_classif_dropout = self.classifier_dropout

        else:
            if self.attention_dropout is not None:
                model.config.attention_probs_dropout_prob = self.attention_dropout
            if self.classifier_dropout is not None:
                model.config.classifier_dropout = self.classifier_dropout

        model.to(self.device)

        return model

    def __init_tokenizer(self) -> AutoTokenizer:
        # bertweet needs normalized inputs
        if self.pretrained_bert_name == "vinai/bertweet-base":
            tokenizer = AutoTokenizer.from_pretrained(self.pretrained_bert_name, normalize=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.pretrained_bert_name)

        return tokenizer

    def tokenize(self, texts: List[str]) -> BatchEncoding:
        tokenized = self.tokenizer(
            texts, truncation=True, padding="max_length", max_length=self.max_seq_len, return_tensors="pt"
        )

        return tokenized

    def __get_dataloader_from_df(self, df: pd.DataFrame) -> DataLoader:
        texts = df.iloc[:, 0].astype("str").to_list()
        targets = df.iloc[:, 1].astype("category").to_list()

        tokenized_train = self.tokenize(texts)
        dataset = BertDataset(tokenized_train, labels=targets)

        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)

        return dataloader

    def __get_optimizer(self, train_dataloader: DataLoader) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        if self.weight_decay is not None:
            no_decay = ["bias", "LayerNorm.weight"]
            parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
        else:
            parameters = self.model.parameters()

        total_steps = len(train_dataloader) * self.num_train_epochs
        num_warmup_steps = int(total_steps * self.warmup_ratio)

        optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate)
        scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
        )

        return optimizer, scheduler

    def __evaluate(self, dev_dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()

        val_accuracy = []
        val_loss = []

        for batch in dev_dataloader:
            batch_inputs = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                output = self.model(**batch_inputs)
                logits = output.logits
                loss = output.loss

            val_loss.append(loss.item())

            preds = torch.argmax(logits, dim=1).flatten()
            labels = batch_inputs["labels"]

            accuracy = (preds == labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        return val_loss, val_accuracy

    def __train(
        self,
        train_dataloader: DataLoader,
        evaluate_during_training=False,
        is_student=False,
        dump_train_history: Optional[bool] = True,
        clip_grad: Optional[bool] = True,
        val_dataloader: Optional[DataLoader] = None,
        unlabeled_dataloader: Optional[DataLoader] = None,
        unl_to_label_batch_ratio: Optional[float] = None,
    ):
        optimizer, scheduler = self.__get_optimizer(train_dataloader)
        progress_bar = tqdm(range(self.num_train_epochs * len(train_dataloader)))
        print_each_n_steps = int(len(train_dataloader) // 4)
        log("Start training...\n")

        historic_loss = {"loss": [], "labeled_loss": [], "unlabeled_loss": [], "steps": [], "unl_steps": []}
        for epoch_i in range(self.num_train_epochs):
            if is_student:
                log(
                    f"{'Epoch':^7} | {'Labeled Batch':^14} | {'Unlabeled Batch':^16} | "
                    f"{'Train Loss':^11} | {'Labeled Loss':^13} | "
                    f"{'Unlabeled Loss':^15} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}"
                )
                log("-" * 130)
            else:
                log(
                    f"{'Epoch':^7} | {'Train Batch':^12} | "
                    f"{'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}"
                )
                log("-" * 80)

            # measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

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
            self.model.train()
            loss_fn = torch.nn.CrossEntropyLoss()
            for step, batch in enumerate(train_dataloader):
                batch_counts += 1
                batch_inputs = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad()
                output = self.model(**batch_inputs)
                # if model is student, train with the noised data aswell
                if is_student:
                    text_col = "augmented_text" if self.has_augmentation else "text"  # TODO: improve this
                    unl_logits = []
                    unl_labels = []

                    for _ in range(unl_to_label_batch_ratio):
                        unl_batch = next(iter(unlabeled_dataloader))

                        unl_texts = unl_batch[text_col]
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

                if (step % print_each_n_steps == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    if is_student:
                        log(
                            f"{epoch_i + 1:^7} | {step:^14} | {(step*unl_to_label_batch_ratio):^16} | "
                            f"{batch_loss / batch_counts:^11.6f} | "
                            f"{batch_lab_loss / batch_counts:^15.6f} | "
                            f"{batch_unl_loss / batch_counts :^13.6f} | "
                            f"{'-':^10} | {'-':^9} | {time_elapsed:^9.2f}"
                        )

                    else:
                        log(
                            f"{epoch_i + 1:^7} | {step:^12} | {batch_loss / batch_counts:^12.6f} | "
                            f"{'-':^10} | {'-':^9} | {time_elapsed:^9.2f}"
                        )

                    batch_loss, batch_lab_loss, batch_unl_loss, batch_counts = 0, 0, 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)
            if evaluate_during_training:
                val_loss, val_accuracy = self.__evaluate(self.model, val_dataloader)
                time_elapsed = time.time() - t0_epoch

                if is_student:
                    log("-" * 130)
                    log(
                        f"{epoch_i + 1:^7} | {'-':^14} | {'-':^16} | {avg_train_loss:^11.6f} | "
                        f"{'-':^15} | {'-':^13}| {val_loss:^10.6f} | "
                        f"{val_accuracy:^9.2f} | {time_elapsed:^9.2f}"
                    )
                    log("-" * 130)
                else:
                    log("-" * 80)
                    log(
                        f"{epoch_i + 1:^7} | {'-':^12} | {avg_train_loss:^12.6f} | "
                        f"{val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}"
                    )
                    log("-" * 80)
            log("\n")

            historic_loss["loss"].append(loss_list)
            historic_loss["labeled_loss"].append(lab_loss_list)
            historic_loss["unlabeled_loss"].append(unl_loss_list)
            historic_loss["unl_steps"].append(unl_step_list)
            historic_loss["steps"].append(step_list)

        if dump_train_history:
            with open(f"train_history-model{self.num_noisy_iteration}.json") as f:
                json.dump(historic_loss, f)

    def predict_batch(self, dataloader: DataLoader) -> List[np.array, np.array]:
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

    def score(self, test_dataloader: DataLoader, dump_test_history: Optional[bool] = True) -> Tuple[float, Dict]:
        self.model.eval()
        all_logits = []
        true_labels = []
        for batch in test_dataloader:
            true_labels.extend(batch["labels"].detach().cpu().numpy())
            batch_inputs = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                output = self.model(**batch_inputs)
                logits = output.logits
            all_logits.append(logits)

        all_logits = torch.cat(all_logits, dim=0)

        probs = torch.nn.functional.softmax(all_logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        # clf_report = classification_report(true_labels, preds)
        f1 = f1_score(true_labels, preds, average="macro")

        if dump_test_history:
            history = {"y_true": [], "y_pred": [], "logits_0": [], "logits_1": []}
            history["y_true"] = true_labels
            history["y_pred"] = preds.tolist()
            history["logits_0"] = all_logits.detach().cpu().numpy()[:, 0]
            history["logits_1"] = all_logits.detach().cpu().numpy()[:, 1]

            with open(f"test_history-model{self.num_noisy_iteration}.json") as f:
                json.dump(history, f)

        return f1, history

    def fit(self, train_df: pd.DataFrame, dev_df: pd.DataFrame):
        train_dataloader = self.__get_dataloader_from_df(train_df)
        if dev_df is not None:
            dev_dataloader = self.__get_dataloader_from_df(dev_df)
