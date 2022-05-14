import numpy as np
import os
import torch

from torch import nn

from adynorm.datasets import CandidateDataset
from preprocess.mention_preprocessor import MCPreprocessor


class Evaluator:
    def __init__(self, ner_model, entity_classifier, adynorm, adynorm_net, tokenizer, id2label, label2id, dictionary,
                 dictionary_embeddings_path):
        self.ner_model = ner_model
        self.entity_classifier = entity_classifier
        self.adynorm = adynorm
        self.adynorm_net = adynorm_net
        if os.path.exists(dictionary_embeddings_path):
            self.dictionary_embeddings = torch.load(dictionary_embeddings_path)
        else:
            self.dictionary_embeddings = adynorm.get_embeddings(dictionary[:, 0], use_tqdm=True)
            torch.save(self.dictionary_embeddings, dictionary_embeddings_path)
        self.dictionary = dictionary
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.label2id = label2id

    @staticmethod
    def _get_bio_indexes(label_ids):
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

    def _ner_pipeline(self, sentence):
        self.ner_model.eval()
        encoded = self.tokenizer(sentence)
        encoded = {k: torch.tensor(v).view(1, -1) for k, v in encoded.items()}
        with torch.no_grad():
            preds = self.ner_model(**encoded)
        label_ids = preds['logits'][0].argmax(dim=1).numpy()
        bio_idxs = self._get_bio_indexes(label_ids)
        bio_entities = []
        for idxs in bio_idxs:
            i = idxs[0]
            bio_entities.append(self.tokenizer.decode(encoded["input_ids"].flatten().numpy()[i:i + len(idxs)]))
        return bio_entities

    def _entity_classifier_pipeline(self, entities):
        entities_valid = []
        for entity in entities:
            ent = {k: torch.tensor(v).view(1, -1) for k, v in self.tokenizer(entity).items()}
            label = torch.argmax(self.entity_classifier(**ent).logits, dim=1)[0]
            print(f'label: {label}')
            if label == 0:
                continue
            entities_valid.append(entity)
        print(f'entities_valid: {entities_valid}')
        return entities_valid

    def _adynorm_pipeline(self, entities, k, max_length):
        result = {
            'predictions': []
        }
        data = []
        preprocessor = MCPreprocessor(True, True)
        for entity in entities:
            entity = preprocessor(entity)
            mention_embedding = self.adynorm.get_embeddings(np.array([entity]))

            score_matrix = self.adynorm.get_score_matrix(mention_embedding, self.dictionary_embeddings)
            candidate_indexes = self.adynorm.get_candidate_indexes(score_matrix, k)
            score_matrix = score_matrix.squeeze()
            scores_sb = score_matrix[candidate_indexes.squeeze()]

            candidates = self.dictionary[candidate_indexes].squeeze()

            candidate_dataset = CandidateDataset([(entity, '')], self.dictionary, self.tokenizer, k, max_length)
            candidate_dataset.load_dense_idxs(candidate_indexes)

            self.adynorm_net.eval()
            with torch.no_grad():
                scores_rb = nn.Sigmoid()(self.adynorm_net(candidate_dataset[0][0]))
            final_scores = scores_rb.squeeze() + scores_sb
            sorted_args_desc = torch.argsort(final_scores, descending=True)
            candidates = candidates[sorted_args_desc.detach().numpy()]
            for i, candidate in enumerate(candidates):
                data.append(
                    {
                        'rank': str(i + 1),
                        'real_name': entity,
                        'name': candidate[0],
                        'cui': candidate[1]
                    }
                )
        result['predictions'] = data
        return result

    def __call__(self, sentence, k=3, max_length=25):
        recognized_entities = self._ner_pipeline(sentence)
        entities_valid = self._entity_classifier_pipeline(recognized_entities)
        return self._adynorm_pipeline(entities_valid, k, max_length)
