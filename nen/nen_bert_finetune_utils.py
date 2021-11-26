import glob
import os
import torch

import numpy as np

from collections import defaultdict
from copy import deepcopy
from nltk import tokenize
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    BertForMaskedLM,
    get_cosine_schedule_with_warmup
)
from typing import (
    List
)

class AbstarctTitleDataset(Dataset):
    def __init__(
        self,
        data_path: str
    ):
        data = []
        at_files = glob.glob(os.path.join(data_path, "*.txt"))
        for at_file in tqdm(at_files):
            with open(at_file, 'r') as file:
                ats = file.readlines()
            sentences = [line.strip('\n') for line in ats if len(line.strip('\n')) > 0]
            data.extend(sentences)
        self.data = np.array(list(set(data)))

def split_text_into_sentences(text: List[str]) -> List[str]:
    return [sent for line in text for sent in tokenize.sent_tokenize(line)]

class MaskedDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        rand = torch.rand(self.data[idx]['input_ids'].shape)
        mask_arr = rand < 0.15

        mask_arr = (rand < 0.15) * (self.data[idx]['input_ids'] != 101) * (self.data[idx]['input_ids'] != 102)

        selection = torch.flatten((mask_arr).nonzero()).tolist()
        
        result = deepcopy(self.data[idx])
        result['input_ids'][selection] = 103
        return result

class PadSequence:
    def __init__(self, padded_columns, device=torch.device('cpu')):
        self.padded_columns = set(padded_columns)
        self.device = device

    def __call__(self, batch):
        padded_batch = defaultdict(list)
        for example in batch:
            for key, tensor in example.items():
                padded_batch[key].append(tensor)
                
        for key, val in padded_batch.items():
            if key in self.padded_columns:
                padded_batch[key] = torch.nn.utils.rnn.pad_sequence(val, batch_first=True).to(self.device)
        return padded_batch

def train(model, optimizer, loader, scheduler, device):
    model.train()
    losses_tr = []
    for batch in tqdm(loader):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        losses_tr.append(loss.item()) 
    
    return model, optimizer, np.mean(losses_tr)

def val(model, loader, device):
    model.eval()
    losses_val = []
    with torch.no_grad():
        for batch in tqdm(loader):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            losses_val.append(loss.item())
    
    return np.mean(losses_val)

def learning_loop(
    model: BertForMaskedLM,
    epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    optimizer: AdamW,
    device=torch.device('cpu')
):
    losses = {'train': [], 'val': []}
    for epoch in range(1, epochs + 1):
        print(f'#{epoch}/{epochs}:')
        clear_cuda_cache()
        
        model, optimizer, loss = train(model, optimizer, train_loader, scheduler, device)
        losses['train'].append(loss)
        
        if test_loader is not None:
            loss = val(model, test_loader, device)
            losses['val'].append(loss)

        if scheduler:
            scheduler.step(loss)

    return model, optimizer, losses