import torch.nn as nn

from transformers import (
    AutoModel
)

class AdynormNet(nn.Module):
    def __init__(self, encoder: AutoModel):
        super().__init__()
        self.encoder = encoder
        self.classification = nn.Linear(768, 1)
    
    def forward(self, pairwise_tokens):
        pairwise_embeddings = self.encoder(
            input_ids=pairwise_tokens['input_ids'].squeeze(1),
            token_type_ids=pairwise_tokens['token_type_ids'].squeeze(1),
            attention_mask=pairwise_tokens['attention_mask'].squeeze(1)
        )
        pairwise_embeddings_cls = pairwise_embeddings[0][:, 0].squeeze(1)
        true_logit = self.classification(pairwise_embeddings_cls)
        return true_logit