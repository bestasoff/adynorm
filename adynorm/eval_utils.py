import numpy as np
import torch

from torch import nn
from tqdm import tqdm

from datasets import CandidateDataset


def get_label(cui, real_cui):
    return int(len(set(cui.split('|')).intersection(set(real_cui.split('|')))) > 0)


def predict(adynorm, model, dictionary, concepts, k, max_length):
    tokenizer = adynorm.get_tokenizer()

    dictionary_embeddings = adynorm.get_embeddings(dictionary[:, 0], use_tqdm=True)
    preds = []

    for concept in tqdm(concepts):
        mentions = concept[0].split('|')
        real_cuis = concept[1]

        dictionary_mentions = []
        for mention in mentions:
            mention_embedding = adynorm.get_embeddings(np.array([mention]))

            score_matrix = adynorm.get_score_matrix(mention_embedding, dictionary_embeddings)
            candidate_indexes = adynorm.get_candidate_indexes(score_matrix, k)
            score_matrix = score_matrix.squeeze()
            scores_sb = score_matrix[candidate_indexes.squeeze()]

            candidates = dictionary[candidate_indexes].squeeze()

            candidate_dataset = CandidateDataset([(mention, '')], dictionary, tokenizer, k, max_length)
            candidate_dataset.load_dense_idxs(candidate_indexes)

            model.eval()
            with torch.no_grad():
                scores_rb = nn.Sigmoid()(model(candidate_dataset[0][0]))
            final_scores = scores_rb.squeeze() + scores_sb
            sorted_args_desc = torch.argsort(final_scores, descending=True)
            candidates = candidates[sorted_args_desc.detach().numpy()]
            dictionary_candidates = []
            for i, candidate in enumerate(candidates):
                dictionary_candidates.append(
                    {
                        'name': candidate[0],
                        'cui': candidate[1],
                        'label': get_label(candidate[1], real_cuis),
                        'score_sb': scores_sb[sorted_args_desc[i]],
                        'score_rb': scores_rb[sorted_args_desc[i]].item(),
                        'score': final_scores[sorted_args_desc[i]].item()
                    }
                )
            dictionary_mentions.append(
                {
                    'mention': mention,
                    'real cui': real_cuis,
                    'candidates': dictionary_candidates
                }
            )
        preds.append(
            {
                'mentions': dictionary_mentions
            }
        )
    return {'preds': preds}


def evaluate_k_acc(output):
    preds = output['preds']
    k = [1, 3, 5, 10]
    for i in k:
        hit = 0
        for pred in preds:
            mentions = pred['mentions']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][: i + 1]
                mention_hit += np.any([candidate['label'] for candidate in candidates])

            if mention_hit == len(mentions):
                hit += 1

        output['acc{}'.format(i + 1)] = hit / len(preds)

    return output


def evaluate(adynorm, model, dictionary, mentions, k, max_length):
    output = predict(adynorm, model, dictionary, mentions, k, max_length)
    result = evaluate_k_acc(output)

    return result
