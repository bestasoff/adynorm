import argparse
import torch

from adynorm import (
	Adynorm
)
from nen_datasets import (
	CandidateDataset,
	ConceptDataset,
	DictDataset
)
from nen_model import (
	AdynormNet
)
from nen_train_utils import learning_loop, criterion
from torch.utils.data import DataLoader

def train(
	train_dict_path: str, 
	val_dict_path: str, 
	processed_train_path: str, 
	processed_val_path: str, 
	model_name_or_path: str, 
	tok_name_or_path: str,
	max_length: int,
	k: int, 
	batch_size: int,
	trained_model_path: str,
	trained_encoder_tokenizer_path: str,
	losses_path: str = None,
	device_name: str = 'cpu',
	epochs: int = 2
	):
	device = torch.device(device_name)
	train_dictionary = DictDataset(train_dict_path).data
	val_dictionary = DictDataset(val_dict_path).data

	train_concept_dataset = ConceptDataset(
	    data_path=processed_train_path,
	    filter_duplicates=True,
	    filter_without_cui=True,
	    filter_composite_names=True
	).data

	val_concept_dataset = ConceptDataset(
	    data_path=processed_val_path,
	    filter_duplicates=True,
	    filter_without_cui=True,
	    filter_composite_names=True
	).data

	train_dictionary_names = train_dictionary[:,0]
	train_concept_names = train_concept_dataset[:,0]

	val_dictionary_names = val_dictionary[:,0]
	val_concept_names = val_concept_dataset[:,0]

	adynorm = Adynorm(
	    max_length=max_length,
	    device=device
	)
	adynorm.load(model_name_or_path, tok_name_or_path)

	model = AdynormNet(encoder=adynorm.get_encoder()).to(device)

	train_dataset = CandidateDataset(
	    concepts=train_concept_dataset, 
	    dictionary=train_dictionary, 
	    tokenizer=adynorm.get_tokenizer(), 
	    k=k, 
	    max_length=max_length
	)

	val_dataset = CandidateDataset(
	    concepts=val_concept_dataset, 
	    dictionary=val_dictionary, 
	    tokenizer=adynorm.get_tokenizer(), 
	    k=k, 
	    max_length=max_length
	)

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	optimizer = torch.optim.Adam(model.parameters(), 5e-8, (0.9, 0.999), weight_decay=1e-2)
	beta = 0.35
	model, optimizer, losses = learning_loop(
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
	    beta=beta,
	    epochs=60,
	    device=device
	)
	torch.save(masked_biobert.to('cpu').state_dict(), finetuned_model_path)
	if losses_path:
		with open(losses_path, 'w') as file:
			json.dumps(losses, file)
	adynorm.save(trained_encoder_tokenizer_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dict_path', type=str, required=True,
    					help='Path to train dictionary.')

    parser.add_argument('--val_dict_path', type=str, required=True,
    					help='Path to val dictionary.')

    parser.add_argument('--processed_train_path', type=str, required=True,
    					help='Path of processed_train.')

    parser.add_argument('--processed_val_path', type=str, required=True,
    					help='Path of processed_val.')

    parser.add_argument('--model_name_or_path', type=str, required=True,
    					help='Name or path to BERT model.')

    parser.add_argument('--tok_name_or_path', type=str, required=True,
    					help='Name or path to tokenizing model.')

    parser.add_argument('--trained_model_path', type=str, required=True,
    					help='Path to save trained model to.')

    parser.add_argument('--trained_encoder_tokenizer_path', type=str, required=True,
                        help='Path to save trained encoder and tokenizer.')

    parser.add_argument('--losses_path', type=str, default=None,
    					help='Path to save train and val losses during training.')

    parser.add_argument('--device', type=str, default='cpu', 
    					help='Device name or cpu.') 

    parser.add_argument('--epochs', type=int, default=2,
    					help='Number of epochs.')

    parser.add_argument('--max_length', type=int, default=2,
    					help='Max length of sequence to feed to BERT.')

    parser.add_argument('--k', type=int, default=2, 
    					help='Number of candidates to retrieve.')

    parser.add_argument('--batch_size', type=int, default=2, 
    					help='Batch size during training.')

    args = parser.parse_args()

    train(
    	args.train_dict_path,
    	args.val_dict_path,
    	args.processed_train_path,
    	args.processed_val_path,
    	args.model_name_or_path,
    	args.tok_name_or_path,
    	args.max_length,
    	args.k,
    	args.batch_size,
    	args.trained_model_path,
    	args.trained_encoder_tokenizer_path,
    	args.losses_path,
    	args.device if args.device else 'cpu',
    	args.epochs
   	)

if __name__ == "__main__":
    main()