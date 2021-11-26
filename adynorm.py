import torch

import numpy as np

from nen_datasets import (
    EmbeddingDataset
)
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    default_data_collator
)
from tqdm import tqdm
from typing import (
    List,
    Sequence
)

class Adynorm:
    def __init__(self, max_length: int, device=torch.device('cpu')):
        self.max_length = max_length
        self.device = device
        
        self.tokenizer: AutoTokenizer = None
        self.encoder: AutoModel = None
            
    def get_encoder(self) -> AutoModel:
        return self.encoder
    
    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer
    
    def save(self, mod_path: str, tok_path: str) -> None:
        self.encoder.to('cpu').save_pretrained(mod_path)
        self.tokenizer.save_pretrained(tok_path)
    
    def load(self, path: str) -> 'Adynorm':
        self.encoder = AutoModel.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

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
        ).to(self.device)
        
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
                embeddings.append(output[0][:,0].cpu().detach().numpy())
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