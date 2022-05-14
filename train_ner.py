import argparse

from torch.optim import AdamW
from torch.utils.data import DataLoader

import logging

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    set_seed,
)

from ner.utils import (
    NerDataset, Split, PadSequence
)
from ner.train_utils import learning_loop

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for NER training.')

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

    labels = ['B', 'I', 'O']
    labels = [l + "-bio" if l != 'O' else l for l in labels]
    print(labels)
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}
    num_labels = len(labels)

    device = 'cuda'

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        config=config
    ).to(device)
    print()
    if args.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()

    train_dataset = (
        NerDataset(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=args.max_seq_length,
            label2id=label2id,
            overwrite_cache=args.overwrite_cache,
            mode=Split.train,
        )
    )
    val_dataset = (
        NerDataset(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=args.max_seq_length,
            label2id=label2id,
            overwrite_cache=args.overwrite_cache,
            mode=Split.val,
        )
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                  collate_fn=PadSequence(['input_ids', 'token_type_ids', 'attention_mask', 'labels']))
    val_dataloader = DataLoader(val_dataset, batch_size=args.train_batch_size, shuffle=False,
                                collate_fn=PadSequence(['input_ids', 'token_type_ids', 'attention_mask', 'labels']))

    optimizer = AdamW(model.parameters(), lr=args.lr)

    _ = learning_loop(
        model,
        args.num_epochs,
        optimizer,
        None,
        train_dataloader,
        val_dataloader,
        val_dataset,
        tokenizer,
        args.accum_steps,
        id2label,
        device,
        save_losses='ner_training_losses.pkl',
        save_metrics='ner_val_metrics.pkl')


if __name__ == "__main__":
    main()
