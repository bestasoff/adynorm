import json
import torch

from ner_utils import (
	get_dataset,
	get_labels_and_label2id,
	tokenize_and_preserve_tags_in_dataset,
	PadSequence,
	NamedEntityPredictor,
	PadSequence,
	AutoConfig,
	AutoTokenizer,
	AutoModelForTokenClassification
)

from ner_train_utils import (
	get_dummy_model_optimizer,
	learning_loop
)

from torch.utils.data import test_dataloader

from seqeval.metrics import classification_report

def train_ner(model_name_or_path: str,
			  input_data_path: str,
			  labels_dir: str, 
			  trained_model_name: str, 
			  lowercase: bool, 
			  remove_punct: bool,
			  device_name: str,
			  epochs: int,
			  losses_path: str = None
			 ):
	labels, label2id = get_labels_and_label2id(labels_dir)
	dataset = get_dataset(data_dir=input_data_path, label2id=label2id)

	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={v: k for k, v in label2id.items()},
        label2id=label2id
    )

    tokenized_dataset = tokenize_and_preserve_tags_in_dataset(dataset, tokenizer, label2id)

    device = torch.device(device_name)

    train_dataloader = DataLoader(
    	tokenized_dataset["train_dev"],
    	batch_size=32,
    	collate_fn=PadSequence(['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    )
	test_dataloader = DataLoader(
		tokenized_dataset["test"],
		batch_size=32,
		collate_fn=PadSequence(['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
	)

	model, optimizer = get_dummy_model_optimizer(model_name_or_path, device, 1e-5)

	scheduler = get_scheduler(
		'cosine',
		optimizer=optimizer,
		num_warmup_steps=0,
		num_training_steps=epochs * len(train_dataloader)
	)

	model, optimizer, losses = learning_loop(
	    model=model,
	    epochs=epochs,
	    optimizer=optimizer,
	    train_loader=train_dataloader,
	    test_loader=test_dataloader,
	    scheduler=scheduler,
	    device=device
	)

	ner = NamedEntityPredictor(model, tokenizer)
	predicted_labels = []

	for batch in tqdm(test_dataloader):
	    predicted_labels.extend(ner.predict(batch)["predicted_labels"])

	print(
		classification_report(y_true=[list(example["text_labels"]) for example in tokenized_dataset["test"]],
                              y_pred=predicted_labels
        )
	)

	torch.save(model.to('cpu').state_dict(), trained_model_name)
	if losses_path:
		with open(losses_path, 'w') as file:
			json.dumps(losses, file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str, required=True,
    					help='BERT name or path.')

    parser.add_argument('--input_data_path', type=str, required=True,
                        help='Path to marked NER dataset')

    parser.add_argument('--trained_model_name', type=str, required=True,
                        help='Name to save trained model')

    parser.add_argument('--labels_dir', type=str, default=None,
                        help='Path to labels file.')

    parser.add_argument('--device', type=str, default='cpu', help='Device name or cpu')

    parser.add_argument('--losses_path', type=str, default=None, help='Path to save train and val losses during training.')

    parser.add_argument('--lowercase',  action="store_true")

    parser.add_argument('--remove_punct',  action="store_true")

    parser.add_argument('--epochs', type=int, default=2, required=True, help='Number of epochs')

    args = parser.parse_args()

    train_ner(
    	args.model_name_or_path,
    	args.input_data_path,
    	args.labels_dir,
    	args.trained_model_name,
    	args.lowercase,
    	args.remove_punct,
    	args.device,
    	args.epochs,
    	args.losses_path
   	)

if __name__ == "__main__":
    main()
