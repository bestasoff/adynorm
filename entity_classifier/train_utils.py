import logging
import pickle

from collections import defaultdict
from tqdm import tqdm
from ner.utils import clear_cuda_cache

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def train(model, optimizer, loader, accum_steps, device):
    model.train()
    losses_tr = [0]
    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / accum_steps
        losses_tr[-1] += loss.item()

        loss.backward()

        if (i + 1) % accum_steps == 0:
            optimizer.step()
            losses_tr.append(0)
            optimizer.zero_grad()
            clear_cuda_cache()

    return model, optimizer, np.mean(losses_tr)


def val(model, loader, dataset, tokenizer, device):
    model.eval()
    losses_val = []
    accum_metrics = defaultdict(int)
    n = len(loader)

    labels_true = []
    labels_pred = []

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            outputs_pred = torch.argmax(outputs.logits, dim=1)
            labels_true.append(labels.detach().cpu().view(-1).numpy())
            labels_pred.append(outputs_pred.detach().cpu().view(-1).numpy())

            loss = outputs.loss
            losses_val.append(loss.item())

    metrics = {
        'precision': precision_score(y_true=np.concatenate(labels_true), y_pred=np.concatenate(labels_pred)),
        'recall': recall_score(y_true=np.concatenate(labels_true), y_pred=np.concatenate(labels_pred)),
        'f1': f1_score(y_true=np.concatenate(labels_true), y_pred=np.concatenate(labels_pred))
    }

    return np.mean(losses_val), metrics


def learning_loop(
        model, num_epochs, optimizer,
        scheduler, train_dataloader, val_dataloader,
        val_dataset, tokenizer, accum_steps,
        device, save_model_steps=10,
        save_losses: str = None, save_metrics: str = None):
    losses = {'train': [], 'val': []}
    val_metrics = {'precision': [], 'recall': [], 'f1': []}
    logger.info(f"*** Learning loop ***")
    for epoch in range(1, num_epochs + 1):
        clear_cuda_cache()

        logger.info(f"*** Train epoch #{epoch} started ***")
        model, optimizer, loss = train(model, optimizer, train_dataloader, accum_steps, device)
        losses['train'].append(loss)
        logger.info(f"*** Train epoch #{epoch} loss *** = {loss}")

        if val_dataloader is not None:
            logger.info(f"*** Validation epoch #{epoch} started ***")
            loss, metrics = val(model, val_dataloader, val_dataset, tokenizer, device)
            for k, i in metrics.items():
                val_metrics[k].append(i)

            losses['val'].append(loss)
            logger.info(
                f"*** Validation epoch #{epoch} results ***\nloss = {loss},\nprecision = {metrics['precision']},\nrecall = {metrics['recall']},\nf1 = {metrics['f1']}")

        if (epoch + 1) % save_model_steps == 0:
            torch.save(model.state_dict(), f'ner/trained_ner_models/model_{epoch}.ct')
            torch.save(optimizer.state_dict(), f'ner/trained_ner_models/optimizer_{epoch}.ct')

        if len(losses['train']) > 2 and abs(losses['train'][-1] - losses['train'][-2]) < 1e-5:
            break

    torch.save(model.state_dict(), f'entity_classifier//trained_models/model_{num_epochs}.ct')
    torch.save(optimizer.state_dict(), f'entity_classifier//trained_models/optimizer_{num_epochs}.ct')

    if save_losses is not None:
        with open(save_losses, 'wb') as file:
            pickle.dump(losses, file)
    if save_metrics is not None:
        with open(save_metrics, 'wb') as file:
            pickle.dump(val_metrics, file)

    return model, optimizer, losses


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

