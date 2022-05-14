import glob
import numpy as np
import os
import torch

from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Tuple, Sequence, Any


class ConceptDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            filter_duplicates: bool,
            filter_without_cui: bool,
            filter_composite_names: bool
    ):
        data = []
        concept_files = glob.glob(os.path.join(data_path, "*.concept"))
        for concept_file in tqdm(concept_files):
            with open(concept_file, 'r') as file:
                concepts = file.readlines()
            for concept in concepts:
                concept = concept.split('||')
                if concept[-2].strip().count('|') > 0 and filter_composite_names:
                    continue
                if concept[-1].strip() == '-1' and filter_without_cui:
                    continue
                data.append((concept[-2].strip(), concept[-1].strip()))
        self.data = np.array(list(set(data))) if filter_duplicates else np.array(data)


class DictDataset(Dataset):
    def __init__(self, dict_path: str):
        data = []
        with open(dict_path, 'r') as file:
            dictionary = file.readlines()
        for record in dictionary:
            record = record.strip()
            if len(record) == 0:
                continue

            cui, mention = record.split('||')
            data.append((mention, cui))
        self.data = np.array(data)


class CandidateDataset(Dataset):
    def __init__(
            self,
            concepts: List[Tuple[str, str]],
            dictionary: List[Tuple[str, str]],
            tokenizer: AutoTokenizer,
            k: int,
            max_length: int
    ):
        self.concepts = np.array([concept[0] for concept in concepts])
        self.concept_cuis = np.array([concept[1] for concept in concepts])
        self.dict_mentions = np.array([concept[0] for concept in dictionary])
        self.dict_mention_cuis = np.array([concept[1] for concept in dictionary])

        self.k = k
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.dense_idxs = None

    def load_dense_idxs(self, dense_idxs: np.array) -> None:
        self.dense_idxs = dense_idxs

    def __getitem__(self, index: int) -> Tuple[Tuple[Sequence[str], Sequence[str]], int]:
        concepts = [self.concepts[index] for _ in range(self.k)]
        candidates = self.dense_idxs[index]
        candidate_names = [self.dict_mentions[idx] for idx in candidates]

        pairwise_tokens = self.tokenizer(
            concepts,
            candidate_names,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            max_length=self.max_length
        )
        labels = self.get_labels(index, candidates)

        return pairwise_tokens, labels

    def get_labels(self, index: int, candidates: np.array) -> np.array:
        labels = []
        cui = self.concept_cuis[index]
        candidate_cuis = self.dict_mention_cuis[candidates]
        for c_cui in candidate_cuis:
            labels.append(self._get_label(cui, c_cui))
        return torch.tensor(labels)

    def _get_label(self, cui: str, c_cui: str) -> int:
        cuis = cui.split('|')
        label = 0
        for c in cuis:
            if c not in c_cui:
                label = 0
                break
            label = 1
        return label

    def __len__(self):
        return len(self.concepts)


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: Sequence[Any]):
        self.embeddings = embeddings

    def __getitem__(self, index: int) -> Any:
        return {key: torch.tensor(val[index]) for key, val in self.embeddings.items()}

    def __len__(self) -> int:
        return len(self.embeddings['input_ids'])
