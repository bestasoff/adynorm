import logging
import os
import re
import torch

from torch.utils.data import Dataset
from tqdm import tqdm
from ner.utils import get_examples, ner_pipeline, get_bio_indexes

logger = logging.getLogger(__name__)


def create_classifier_dataset(model, tokenizer, id2label, label2id, data_dir, mode, thr=0.65, overwrite_cache=False):
    cached_features_file = os.path.join(
        data_dir, "cached_classifier_features_{}".format(mode),
    )
    if os.path.exists(cached_features_file) and not overwrite_cache:
        logger.info(f"Loading features from cached file {cached_features_file}")
        return torch.load(cached_features_file)

    logger.info(f"Creating features from dataset file at {data_dir}")
    file_path = os.path.join(data_dir, f'{mode}.txt')
    if not os.path.isfile(file_path):
        raise Exception(f'File {file_path} doesn\'t exist.')

    examples = get_examples(data_dir, mode, label2id)
    for i, example in tqdm(enumerate(examples), total=len(examples)):
        examples[i]['predicted_bio_entities'] = ner_pipeline(model, tokenizer, ' '.join(example['words']), id2label)

        bio_idxs = get_bio_indexes(example['ner_tags'])
        bio_entities = []
        for idxs in bio_idxs:
            j = idxs[0]
            bio_entities.append(re.sub(r' \' ', '\'', ' '.join(example['words'][j:j + len(idxs)]).lower()))
        examples[i]['real_bio_entities'] = bio_entities

        labels = []
        for p_b_e in examples[i]['predicted_bio_entities']:
            for b_e in bio_entities:
                s_p_b_e = set(p_b_e.split())
                s_b_e = set(b_e.split())
                iou = len(s_p_b_e & s_b_e) / len(s_p_b_e | s_b_e)
                if iou > thr:
                    labels.append(1)
                    break
            else:
                labels.append(0)
        examples[i]['labels'] = labels

    logger.info(f"Saving features into cached file {cached_features_file}")
    torch.save(examples, cached_features_file)
    return examples


class EntityClassifierDataset(Dataset):
    def __init__(self, dataset_dir, mode, tokenizer):
        dataset_mode_dir = os.path.join(dataset_dir, f'cached_classifier_features_{mode}')
        if not os.path.exists(dataset_mode_dir):
            raise Exception(f"File {dataset_mode_dir} does not exist.")
        dataset = torch.load(dataset_mode_dir)

        self.tokenizer = tokenizer
        self.entities = []

        for e in tqdm(dataset):
            for i, entity in enumerate(e['predicted_bio_entities']):
                ent = {k: torch.tensor(v) for k, v in self.tokenizer(entity).items()}
                ent['labels'] = torch.tensor([e['labels'][i]])
                self.entities.append(ent)

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, idx):
        return self.entities[idx]
