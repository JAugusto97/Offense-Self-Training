import os
import random
import time
from datetime import datetime

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AdamW, get_scheduler


def plog(text):
    print(text)
    log.write(text + "\n")

def load_olid(datasets_path):
    eng_path = os.path.join(datasets_path, "english")

    train_path = os.path.join(eng_path, "OLIDv1.0", "olid-training-v1.0.tsv")
    test_path = os.path.join(eng_path, "OLIDv1.0", "testset-levela.tsv")
    test_labels_path = os.path.join(eng_path, "OLIDv1.0", "labels-levela.csv")
    unlabeled_path = os.path.join(eng_path, "unlabeled", "tweets_augmented.csv")

    train_df = pd.read_csv(train_path, engine="python", sep='\t')[["tweet", "subtask_a"]]
    train_df["subtask_a"] = train_df["subtask_a"].apply(lambda x: 1 if x == "OFF" else 0)
    train_df = train_df.rename({"tweet": "text", "subtask_a": "toxic"}, axis=1)

    test_df = pd.read_csv(test_path, engine="python", sep='\t')
    test_labels = pd.read_csv(test_labels_path, header=None)
    test_df["toxic"] = test_labels[1].apply(lambda x: 1 if x == "OFF" else 0)
    test_df = test_df[["tweet", "toxic"]]
    test_df = test_df.rename({"tweet": "text"}, axis=1)

    unlabeled_df = pd.read_csv(unlabeled_path)[["text", "text_augmented"]]
    unlabeled_df["text"] = unlabeled_df["text"]
    unlabeled_df["text_augmented"] = unlabeled_df["text_augmented"]

    plog(
        f"""
        Loaded OLID V1.0

        Train Size: {len(train_df)}
            Positives: {len(train_df[train_df["toxic"] == 1])}
            Negatives: {len(train_df[train_df["toxic"] == 0])}
        Test Size: {len(test_df)}
            Positives: {len(test_df[test_df["toxic"] == 1])}
            Negatives: {len(test_df[test_df["toxic"] == 0])}
        Augmented Data: {len(unlabeled_df)}

        """
    )
    return train_df, test_df, unlabeled_df

class OlidDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, df, labels):
        self.text = df["text"].to_list()
        self.labels = labels
        self.text_augmented = df["text_augmented"].to_list()

    def __getitem__(self, idx):
        item = {"text": self.text[idx], "labels": self.labels[idx], "text_augmented": self.text_augmented[idx]}
        return item

    def __len__(self):
        return len(self.labels)

class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, pretrained_bert_name, hidden_dim, n_classes, dropout_prob=0.1):
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, hidden_dim, n_classes

        # Instantiate BERT model
        self.bert = AutoModel.from_pretrained(pretrained_bert_name)

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(H, D_out)
        )
    
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits

def initialize_model(pretrained_bert_name, hidden_dim, n_labels, dropout_proba, epochs):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    model = BertClassifier(pretrained_bert_name, hidden_dim, n_labels, dropout_proba)

    # Create the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    return model, optimizer, scheduler

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(
    model,
    train_dataloader,
    epochs,
    optimizer,
    scheduler,
    val_dataloader=None,
    evaluate_during_training=False,
    is_student=False,
    unlabeled_dataloader=None,
):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()


            # Perform a forward pass. This will return logits.
            logits = model(input_ids, attention_mask)

            if is_student:
                unl_batch = next(iter(unlabeled_dataloader))
                unl_inputs = batch_tokenize(unl_batch["text_augmented"])

                unl_input_ids = torch.LongTensor(unl_inputs['input_ids']).to(device)
                unl_attention_mask = torch.LongTensor(unl_inputs['attention_mask']).to(device)
                unl_labels = unl_batch["labels"].to(device)

                unl_logits = model(unl_input_ids, unl_attention_mask)

                loss = loss_fn(logits, labels)
                loss = loss + loss_fn(unl_logits, unl_labels)

            else:
                loss = loss_fn(logits, labels)

            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluate_during_training:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    
    print("Training complete!")


def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Compute logits
        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        # Compute loss
        loss = loss_fn(logits, labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

def bert_predict(model, dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    all_logits = []

    # For each batch in our test set...
    for batch in dataloader:
        # Load batch to GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Compute logits
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    labels = np.argmax(probs, axis=1)

    return probs, labels

def batch_tokenize(sents):
    tokenized = tokenizer.batch_encode_plus(
        sents,
        padding="max_length",
        truncation=True,
        max_length=CFG.max_seq_len
    )

    return tokenized
