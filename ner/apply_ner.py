from ner_utils import get_eval_dataset

def evaluate(path_to_data: str, model_name_or_path: str, output_path: str, labels_dir: str):
	labels, label2id = get_labels_and_label2id(labels_dir)
	dataset = get_eval_dataset(data_dir)

	tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=output_dir
    )
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={v: k for k, v in label2id.items()},
        label2id=label2id,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        config=config,
    )

    tokenized_dataset = [tokenizer(sent[:512]) for sent in tqdm(dataset)]

    test_dataloader = DataLoader(
    	tokenized_dataset["test"], 
    	batch_size=4, 
    	collate_fn=PadSequence(['input_ids', 'token_type_ids', 'attention_mask'])
    )

    ner = NamedEntityPredictor(model, tokenizer)
	predicted_labels = []

	for batch in tqdm(test_dataloader):
	    predicted_labels.extend([(b, n) for b, n in zip(batch, ner.predict(batch)["predicted_labels"])])

	with open(output_path, 'w') as file:
		json.dump(predicted_labels, file)
		
# TODO finish evaluation