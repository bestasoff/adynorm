from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from seqeval.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import torch


class NamedEntityPredictor:
    def __init__(self,
                 model: AutoModelForTokenClassification,
                 tokenizer: AutoTokenizer,
                 id2label, device):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = model.config.id2label if id2label is None else id2label
        self.device = device

    def predict(self, batch):
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(
                input_ids=batch["input_ids"].to(self.device),
                token_type_ids=batch["token_type_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                labels=batch["labels"].to(self.device),
                return_dict=True
            )
        indices = torch.argmax(model_output.logits, dim=2)
        indices = indices.detach().cpu().numpy()
        attention_mask = batch["attention_mask"].cpu().numpy()
        batch_size = len(batch["input_ids"])
        predicted_labels = []
        for i in range(batch_size):
            predicted_labels.append([self.id2label[id_] for id_ in indices[i][attention_mask[i] == 1]])

        return {
            "predicted_labels": predicted_labels,
            "loss": model_output.loss,
            "logits": model_output.logits
        }


def get_metrics(model, tokenizer, device, loader, dataset, id2label):
    ner = NamedEntityPredictor(model, tokenizer, id2label, device)
    predicted_labels = []

    for batch in tqdm(loader):
        predicted_labels.extend(ner.predict(batch)["predicted_labels"])

    y_true = [list(example["text_labels"]) for example in dataset]
    y_pred = predicted_labels
    # assert len(y_true) == len(y_pred)
    # print(classification_report(y_true=y_true, y_pred=y_pred, output_dict=True))

    return {
        'precision': precision_score(y_true=y_true, y_pred=y_pred),
        'recall': recall_score(y_true=y_true, y_pred=y_pred),
        'f1': f1_score(y_true=y_true, y_pred=y_pred)
    }
