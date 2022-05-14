import argparse

from torch.optim import AdamW
from torch.utils.data import DataLoader

import logging

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)

from entity_classifier.train_utils import (
    PadSequence, learning_loop
)
from entity_classifier.utils import EntityClassifierDataset

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for entity classifier training.')

    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)

    parser.add_argument('--model_name_or_path', type=str, default='dmis-lab/biobert-base-cased-v1.1')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--accum_steps', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--overwrite_output_dir', action="store_true")
    parser.add_argument('--overwrite_cache', action="store_true")
    parser.add_argument('--gradient_checkpointing_enable', action="store_true")

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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
    ).to(device)

    train_dataset = EntityClassifierDataset('../datasets_new/NCBI/preprocessed_ner/', 'train', tokenizer)
    val_dataset = EntityClassifierDataset('../datasets_new/NCBI/preprocessed_ner/', 'val', tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                  collate_fn=PadSequence(['input_ids', 'token_type_ids', 'attention_mask', 'labels']))
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                                collate_fn=PadSequence(['input_ids', 'token_type_ids', 'attention_mask', 'labels']))

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = None

    _ = learning_loop(
        model,
        args.num_epochs,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        val_dataset,
        tokenizer,
        args.accum_steps,
        device,
        save_losses='classifier_training_losses.pkl',
        save_metrics='classifier_val_metrics.pkl')


if __name__ == "__main__":
    main()
