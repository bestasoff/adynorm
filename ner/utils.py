import logging
import os
import torch
import gc

from enum import Enum
from typing import List

from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset


class Split(Enum):
    train = "train"
    val = "val"
    test = "test"


logger = logging.getLogger(__name__)


def get_examples(data_dir: str, mode: str, label2id: dict):
    file_path = os.path.join(data_dir, f"{mode}.txt")
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as file:
        words = []
        labels = []
        for line in file:
            if line == "" or line == "\n":
                if words:
                    examples.append({
                        "ner_tags": [label2id[label] for label in labels],
                        "words": words,
                        "ner": labels,
                        "id": f"{mode}-{guid_index}",
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
                        # labels.append(label)
                        labels.append(label + "-bio")
                else:
                    labels.append("O")
        if words:
            examples.append({
                "ner_tags": [label2id[label] for label in labels],
                "words": words,
                "ner": labels,
                "id": f"{mode}-{guid_index}",
                "title": None,
            })
    return examples


def tokenize_and_preserve_tags(example, tokenizer, label2id, tokenizer_params={}):
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


def tokenize_and_preserve_tags_in_dataset(examples, tokenizer, label2id):
    processed_examples = []
    for example in examples:
        processed_examples.append(tokenize_and_preserve_tags(example, tokenizer, label2id))
    return processed_examples


class NerDataset(Dataset):
    features: List[dict]

    def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: int,
            label2id: dict,
            overwrite_cache=False,
            mode: Split = Split.train,
    ):
        cached_features_file = os.path.join(
            data_dir, "cached_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            self.features = torch.load(cached_features_file)
        else:
            logger.info(f"Creating features from dataset file at {data_dir}")
            examples = get_examples(data_dir, mode.value, label2id)
            self.features = tokenize_and_preserve_tags_in_dataset(examples, tokenizer, label2id)
            logger.info(f"Saving features into cached file {cached_features_file}")
            torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


def clear_cuda_cache():
    gc.collect()

    torch.cuda.empty_cache()


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


def get_bio_indexes(label_ids):
    bio_idxs = []
    i = 0
    while i < len(label_ids):
        if label_ids[i] == 0:
            idxs = []
            while i < len(label_ids) and label_ids[i] != 2:
                idxs.append(i)
                i += 1
            bio_idxs.append(idxs)
        else:
            i += 1
    return bio_idxs


def ner_pipeline(model, tokenizer, sentence, id2label):
    model.eval()
    encoded = tokenizer(sentence)
    word_ids = encoded.word_ids()
    encoded = {k: torch.tensor(v).view(1, -1) for k, v in encoded.items()}
    with torch.no_grad():
        preds = model(**encoded)
    label_ids = preds['logits'][0].argmax(dim=1).numpy()
    tokenized = " ".join(tokenizer.convert_ids_to_tokens(encoded["input_ids"].flatten().numpy()))
    bio_idxs = get_bio_indexes(label_ids)
    bio_entities = []
    for idxs in bio_idxs:
        i = idxs[0]
        bio_entities.append(tokenizer.decode(encoded["input_ids"].flatten().numpy()[i:i + len(idxs)]))
    return bio_entities