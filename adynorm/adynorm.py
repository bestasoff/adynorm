import numpy as np
import torch

from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, default_data_collator
from tqdm import tqdm
from typing import List, Sequence

from .datasets import EmbeddingDataset

device = 'cpu'


class Adynorm:
    def __init__(self, max_length: int, device=None):
        self.max_length = max_length
        self.device = device

        self.tokenizer: AutoTokenizer = None
        self.encoder: AutoModel = None

    def get_encoder(self) -> AutoModel:
        return self.encoder

    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer

    def save(self, path: str) -> None:
        self.encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str, tok_path: str) -> 'Adynorm':
        self.encoder = AutoModel.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        if self.device is not None:
            self.encoder.to(device)
        return self

    def get_embeddings(self, mentions: List[str], use_tqdm: bool = False) -> List[Sequence[float]]:
        if isinstance(mentions, np.ndarray):
            mentions = mentions.tolist()
        mentions_tokens = self.tokenizer(
            mentions,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        if device is not None:
            mentions_tokens.to(device)

        self.encoder.eval()
        batch_size = 1024
        embeddings = []

        emb_dataset = EmbeddingDataset(mentions_tokens)
        emb_dataloader = DataLoader(
            emb_dataset,
            shuffle=False,
            collate_fn=default_data_collator,
            batch_size=batch_size
        )

        with torch.no_grad():
            for batch in tqdm(emb_dataloader, disable=not use_tqdm):
                output = self.encoder(**batch)
                #                 embeddings.append(output[0].sum(axis=1).cpu().detach().numpy())
                #                 print(embeddings[-1].shape)
                embeddings.append(output[0][:, 0].cpu().detach().numpy())
        return np.concatenate(embeddings, axis=0)

    def get_candidate_indexes(self, score_matrix: np.array, k: int) -> Sequence[int]:
        def indexing_2d(arr: np.array, cols: np.array):
            rows = np.repeat(np.arange(0, cols.shape[0])[:, np.newaxis], cols.shape[1], axis=1)
            return arr[rows, cols]

        k_idxs = np.argpartition(score_matrix, -k)[:, -k:]
        k_score_matrix = indexing_2d(score_matrix, k_idxs)
        k_argidxs = np.argsort(-k_score_matrix)
        k_idxs = indexing_2d(k_idxs, k_argidxs)

        return k_idxs

    def get_score_matrix(self, mention_embeddings, dictionary_embeddings):
        score_matrix = cosine_similarity(mention_embeddings, dictionary_embeddings)
        return score_matrix


class AdynormNet(nn.Module):
    def __init__(self, encoder: AutoModel):
        super().__init__()
        self.encoder = encoder

        self.classification = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, pairwise_tokens):
        pairwise_embeddings = self.encoder(
            input_ids=pairwise_tokens['input_ids'].squeeze(1),
            token_type_ids=pairwise_tokens['token_type_ids'].squeeze(1),
            attention_mask=pairwise_tokens['attention_mask'].squeeze(1)
        )
        pairwise_embeddings_cls = pairwise_embeddings[0][:, 0].squeeze(1)
        true_logit = self.classification(pairwise_embeddings_cls)
        return true_logit
