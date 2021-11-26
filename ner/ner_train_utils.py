import gc
import os
import torch

import numpy as np
import pandas as pd

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

from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    AdamW,
    AutoModelForTokenClassification,
    AutoModel, 
    AutoTokenizer,
    PreTrainedTokenizer,
    get_scheduler,
    set_seed
)

def clear_cuda_cache():
    gc.collect()

    torch.cuda.empty_cache()

def get_dummy_model_optimizer(model_name_or_path, config, device, lr):
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    return model, optimizer

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
    model: AutoModelForTokenClassification,
    epochs: int,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
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