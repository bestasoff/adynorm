import argparse

import logging

from transformers import (
    set_seed,
)

from adynorm.eval_utils import (
    evaluate
)
from adynorm.adynorm import Adynorm, AdynormNet
from adynorm.datasets import ConceptDataset, DictDataset

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for entity classifier training.')

    parser.add_argument('--processed_val_path', required=True)
    parser.add_argument('--val_dict_path', required=True)
    parser.add_argument('--max_length', type=int, default=25)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--model_name_or_path', type=str, default='dmis-lab/biobert-base-cased-v1.1')

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

    device = 'cpu'

    val_dictionary = DictDataset(args.val_dict_path).data

    val_concept_dataset = ConceptDataset(
        data_path=args.processed_val_path,
        filter_duplicates=True,
        filter_without_cui=True,
        filter_composite_names=True
    ).data

    adynorm = Adynorm(
        max_length=args.max_length,
        device=device
    )

    adynorm.load(args.model_name_or_path, args.model_name_or_path)

    model = AdynormNet(encoder=adynorm.get_encoder()).to(device)

    result = evaluate(adynorm, model.to('cpu'), val_dictionary, val_concept_dataset, 35, args.max_length)

    for k in result.keys():
        if k == 'preds':
            continue
        print(k, result[k])


if __name__ == "__main__":
    main()
