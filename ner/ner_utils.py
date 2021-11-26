import gc
import logging
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import re
import sys
import subprocess
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
)

from aenum import Enum
from filelock import FileLock
from IPython.display import clear_output
from nltk import tokenize
from nltk.corpus import stopwords
from seqeval.metrics import (
    f1_score,
    recall_score,
    precision_score
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModel, 
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer, 
    TrainingArguments,
    default_data_collator,
    get_scheduler,
    set_seed
)

class Split(Enum):
    train = "train_dev"
    dev = "devel"
    test = "test"

def get_dataset(data_dir: str, label2id: Dict[str, int]) -> Dict[str, List[Any]]:
    dataset = {}
    for mode in ["train", "dev", "test"]:
        mode = getattr(Split, mode).value
        
        file_path = os.path.join(data_dir, f"{mode}.txt")
        
        guid_index = 1
        samples = []
        with open(file_path, encoding="utf-8") as file:
            words = []
            labels = []
            for line in file:
                if line == "" or line == "\n":
                    if words:
                        samples.append({
                            "ner_tags": [label2id[label] for label in labels],
                            "words": words,
                            "ner": labels,
                            "id": guid_index,
                            "title": None,
                        })
                        guid_index += 1
                        words = []
                        labels = []
                else:
                    tokens = line.split(" ")
                    words.append(tokens[0])
                    if len(tokens) > 1:
                        label = tokens[-1].replace("\n", "")
                        if label == "O":
                            labels.append(label)
                        else:
                            labels.append(label + "-bio")
                    else:
                        labels.append("O")
            if words:
                samples.append({
                            "ner_tags": [label2id[label] for label in labels],
                            "words": words,
                            "ner": labels,
                            "id": guid_index,
                            "title": None,
                        })
            dataset[mode] = samples
    return dataset

def get_eval_dataset(dataset_path: str) -> List[Any]:
    data = []
    with open(dataset_path, 'r') as file:
            ats = file.readlines()
        sentences = [line.strip('\n') for line in ats if len(line.strip('\n')) > 0]
        data.extend(sentences)
    data = [sent for line in data for sent in tokenize.sent_tokenize(line)]
    return data
        
def get_labels_and_label2id(labels_dir: str) -> List[str]:
    labels = [item.strip() + "-bio" if item.strip() != "O" else item.strip()
              for item in open(labels_dir, "r").readlines()]
    label2id = {label: idx for idx, label in enumerate(labels)}
    return labels, label2id

def ner_render(words: List[str], ner: List[str], title: Optional[str] = None, **kwargs):
    pos = 0
    ents = []
    for word, tag in zip(words, ner):
        if tag.startswith('B'):
            ents.append({
                "start": pos,
                "end": pos + len(word),
                "label": tag.split("-")[1]
            })
        elif tag.startswith('I'):
            ents[-1]["end"] = pos + len(word)
        pos += (len(word) + 1)
    displacy.render({
        "text": " ".join(words),
        "ents": ents,
        "title": title
    }, style="ent", manual=True)

# from spacy import displacy
# ner_render(**dataset['test'][600])

class ModeDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

def tokenize_and_preserve_tags(example: Dict[str, Any],
                               tokenizer: AutoTokenizer,
                               label2id: Dict[str, int],
                               tokenizer_params={}) -> Dict[str, Any]:

    encoded = tokenizer(example["words"], is_split_into_words=True, **tokenizer_params)
    encoded.update(example)
    
    word_ids = encoded.word_ids()
    text_labels = []
    for i in range(1, len(word_ids) - 1):
        label = example["ner"][word_ids[i]]
        if label == "O" or word_ids[i - 1] != word_ids[i]:
            text_labels.append(label)
        else:
            text_labels.append("I" + label[1:])
    text_labels = ["O"] + text_labels + ["O"]
    
    encoded['labels'] = [label2id[label] for label in text_labels]
    encoded['text_labels'] = text_labels
    
    for key in ['labels', 'input_ids', 'attention_mask', 'token_type_ids']:
        encoded[key] = torch.tensor(encoded[key])
    
    assert len(encoded['labels']) == len(encoded["input_ids"])
    return encoded

def tokenize_and_preserve_tags_in_dataset(
    dataset: Dict[str, List[Dict[str, Any]]],
    tokenizer: AutoTokenizer,
    label2id: Dict[str, int]
) -> Dict[str, List[Dict[str, Any]]]:
    new_dataset = {}
    for key in dataset:
        data = dataset[key]
        
        for example in data:
            new_dataset.setdefault(key, []).append(tokenize_and_preserve_tags(example, tokenizer, label2id))
    for key in dataset:
        new_dataset[key] = ModeDataset(new_dataset[key])
    return new_dataset

class PadSequence:
    def __init__(self, padded_columns):
        self.padded_columns = set(padded_columns)

    def __call__(self, batch):
        padded_batch = {}
        for example in batch:
            for key, tensor in example.items():
                padded_batch.setdefault(key, []).append(tensor)
                
        for key, val in padded_batch.items():
            if key in self.padded_columns:
                padded_batch[key] = torch.nn.utils.rnn.pad_sequence(val, batch_first=True)
        return padded_batch

class NamedEntityPredictor:
    def __init__(self,
                 model: AutoModelForTokenClassification,
                 tokenizer: AutoTokenizer,
                 id2label: Optional[Dict[str, int]] = None):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.id2label = model.config.id2label if id2label is None else id2label
    
    def predict(self, batch: Dict[str, Any]):
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(input_ids=batch["input_ids"].to(device),
                                      token_type_ids=batch["token_type_ids"].to(device),
                                      attention_mask=batch["attention_mask"].to(device),
                                      labels=batch["labels"].to(device),
                                      return_dict=True)
        indices = torch.argmax(model_output.logits, axis=2)
        indices = indices.detach().cpu().numpy()
        attention_mask = batch["attention_mask"].cpu().numpy()
        batch_size = len(batch["input_ids"])
        predicted_labels = []
        for i in range(batch_size):
            predicted_labels.append([self.id2label[id_] for id_ in indices[i][attention_mask[i] == 1]])
            
        return {
            "predicted_labels": predicted_labels,
            "loss": model_output.loss,
            "logits": model_output.logits
        }