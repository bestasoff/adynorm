import argparse
import os

from torch.data.utils import DataLoader
from nen_bert_finetune_utils import (
	MaskedDataset,
	PadSequence,
	learning_loop,
	split_text_into_sentences
)
from trasnformers import (
	AdamW,
	BertForMaskedLM,
	get_cosine_schedule_with_warmup
)
from typing import (
	List
)

def finetune(
	path_to_texts: str, 
	model_name_or_path: str, 
	finetuned_model_path: str,
	losses_path: str = None,
	splits: List[str] = None, 
	device_name: str = 'cpu',
	epochs: int = 2
	):
	data = []
	if splits:
		for split in splits:
			data.extend(AbstarctTitleDataset(os.path.join(path_to_texts, split)).data)
	else:
		data.extend(AbstarctTitleDataset(path_to_texts).data)

	device = torch.device(device_name)

	data = split_text_into_sentences(data)

	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

	tokenized_data = [tokenizer(sent[:512]) for sent in tqdm(data)]

	tokenized_data = [{k: torch.tensor(v) for k, v in line.items()} for line in tokenized_data]
	for i in range(len(tokenized_data)):
	    tokenized_data[i]['labels'] = tokenized_data[i]['input_ids'].detach().clone()

	masked_dataset = MaskedDataset(tokenized_data)
	masked_dataloader = DataLoader(
	    masked_dataset,
	    batch_size=8,
	    collate_fn=PadSequence(['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
	)

	masked_biobert = BertForMaskedLM.from_pretrained(model_name_or_path).to(device)

	optimizer = AdamW(masked_biobert.parameters(), lr=3e-6)
	scheduler = get_cosine_schedule_with_warmup(
	    optimizer, num_warmup_steps=7, 
	    num_training_steps=epochs * len(masked_dataloader)
	)

	masked_biobert, optimizer, losses = learning_loop(
	    model=masked_biobert,
	    epochs=epochs,
	    optimizer=optimizer,
	    train_loader=masked_dataloader,
	    test_loader=None,
	    scheduler=scheduler
	)

	torch.save(masked_biobert.to('cpu').state_dict(), finetuned_model_path)
	if losses_path:
		with open(losses_path, 'w') as file:
			json.dumps(losses, file)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str, required=True,
    					help='BERT name or path.')

    parser.add_argument('--path_to_texts', type=str, required=True,
                        help='Path to texts to finetune model.')

    parser.add_argument('--finetuned_model_path', type=str, required=True,
                        help='Path to save finetuned model.')

    parser.add_argument('--device', type=str, default='cpu', help='Device name or cpu')

    parser.add_argument('--losses_path', type=str, default=None, help='Path to save train and val losses during training.')

    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')

    parser.add_argument('--splits', type=str, default=None, help='Splits in texts path. Write them this pattern: split1.split2,split3...')

    args = parser.parse_args()

    finetune(
    	args.path_to_texts,
    	args.model_name_or_path,
    	args.finetuned_model_path,
    	args.losses_path,
    	args.splits.split(',') if args.splits else None,
    	args.device,
    	args.epochs
   	)

if __name__ == "__main__":
    main()
