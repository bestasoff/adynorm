import argparse
import torch

from torch.utils.data import DataLoader

import logging

from transformers import (
    set_seed,
)

from adynorm.train_utils import (
    learning_loop, criterion
)
from adynorm.adynorm import Adynorm, AdynormNet
from adynorm.datasets import ConceptDataset, DictDataset, CandidateDataset

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for entity classifier training.')

    parser.add_argument('--processed_train_path', required=True)
    parser.add_argument('--processed_val_path', required=True)
    parser.add_argument('--train_dict_path', required=True)
    parser.add_argument('--val_dict_path', required=True)
    parser.add_argument('--max_length', type=int, default=25)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--model_name_or_path', type=str, default='dmis-lab/biobert-base-cased-v1.1')

    parser.add_argument('--accum_steps', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("Training/evaluation parameters %s", args)

    set_seed(args.seed)

    device = 'cuda'

    train_dictionary = DictDataset(args.train_dict_path).data
    val_dictionary = DictDataset(args.val_dict_path).data

    train_concept_dataset = ConceptDataset(
        data_path=args.processed_train_path,
        filter_duplicates=True,
        filter_without_cui=True,
        filter_composite_names=True
    ).data

    val_concept_dataset = ConceptDataset(
        data_path=args.processed_val_path,
        filter_duplicates=True,
        filter_without_cui=True,
        filter_composite_names=True
    ).data

    train_dictionary_names = train_dictionary[:, 0]
    train_concept_names = train_concept_dataset[:, 0]

    val_dictionary_names = val_dictionary[:, 0]
    val_concept_names = val_concept_dataset[:, 0]

    adynorm = Adynorm(
        max_length=args.max_length,
        device=device
    )

    adynorm.load(args.model_name_or_path, args.model_name_or_path)

    model = AdynormNet(encoder=adynorm.get_encoder()).to(device)

    train_dataset = CandidateDataset(
        concepts=train_concept_dataset,
        dictionary=train_dictionary,
        tokenizer=adynorm.get_tokenizer(),
        k=args.k,
        max_length=args.max_length
    )

    val_dataset = CandidateDataset(
        concepts=val_concept_dataset,
        dictionary=val_dictionary,
        tokenizer=adynorm.get_tokenizer(),
        k=args.k,
        max_length=args.max_length
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), weight_decay=1e-2)

    _ = learning_loop(
        model=model,
        adynorm=adynorm,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_concept_names=train_concept_names,
        val_concept_names=val_concept_names,
        train_dictionary_names=train_dictionary_names,
        val_dictionary_names=val_dictionary_names,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        criterion=criterion,
        accum_steps=args.accum_steps,
        beta=args.beta,
        epochs=60
    )


if __name__ == "__main__":
    main()
