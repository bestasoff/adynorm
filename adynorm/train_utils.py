import gc
import numpy as np
import torch

from torch import nn
from tqdm import tqdm

device = torch.device('cuda')


def criterion(logits, beta, y):
    probs = nn.Sigmoid()(logits)
    probs = torch.cat([1 - probs, probs], dim=1)
    entropy = (probs * torch.log(probs)).mean()
    nll = torch.log(probs.gather(1, y.unsqueeze(1))).mean()
    loss = -beta * entropy - nll
    return loss


def clear_cuda_cache():
    gc.collect()
    torch.cuda.empty_cache()


def train(model, optimizer, loader, beta, criterion, accum_steps, max_length):
    model.train()
    losses_tr = []
    for i, (X, y) in tqdm(enumerate(loader), total=len(loader)):
        X['input_ids'] = X['input_ids'].reshape(-1, max_length).to(device)
        X['attention_mask'] = X['attention_mask'].reshape(-1, max_length).to(device)
        X['token_type_ids'] = X['token_type_ids'].reshape(-1, max_length).to(device)

        y = y.flatten().to(device)

        output = model(X)
        loss = criterion(output, beta, y)

        loss.backward()

        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            clear_cuda_cache()

        losses_tr.append(loss.item())

    return model, optimizer, np.mean(losses_tr)


def val(model, loader, beta, criterion, max_length):
    model.eval()
    losses_val = []
    with torch.no_grad():
        for X, y in tqdm(loader):
            X['input_ids'] = X['input_ids'].reshape(-1, max_length).to(device)
            X['attention_mask'] = X['attention_mask'].reshape(-1, max_length).to(device)
            X['token_type_ids'] = X['token_type_ids'].reshape(-1, max_length).to(device)

            y = y.flatten().to(device)

            output = model(X)
            loss = criterion(output, beta, y)

            losses_val.append(loss.item())

    return np.mean(losses_val)


def learning_loop(
        model,
        adynorm,
        optimizer,
        train_dataset,
        val_dataset,
        train_concept_names,
        val_concept_names,
        train_dictionary_names,
        val_dictionary_names,
        train_loader,
        val_loader,
        criterion,
        beta=0.5,
        scheduler=None,
        epochs=20,
        k=20,
        accum_steps=2,
        use_tqdm=True
):
    losses = {'train': [], 'val': []}

    loss = 0
    for epoch in range(1, epochs + 1):
        clear_cuda_cache()
        print(f'#{epoch}/{epochs}:')
        print(f"Loss: {loss}")

        train_concepts_embeds = adynorm.get_embeddings(mentions=train_concept_names, use_tqdm=use_tqdm)
        train_dictionary_embeds = adynorm.get_embeddings(mentions=train_dictionary_names, use_tqdm=use_tqdm)

        train_score_matrix = adynorm.get_score_matrix(
            mention_embeddings=train_concepts_embeds,
            dictionary_embeddings=train_dictionary_embeds
        )
        train_candidate_idxs = adynorm.get_candidate_indexes(
            score_matrix=train_score_matrix,
            k=k
        )
        train_dataset.load_dense_idxs(train_candidate_idxs)

        model, optimizer, loss = train(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            criterion=criterion,
            beta=beta,
            accum_steps=accum_steps
        )
        losses['train'].append(loss)

        clear_cuda_cache()

        val_concepts_embeds = adynorm.get_embeddings(mentions=val_concept_names, use_tqdm=use_tqdm)
        val_dictionary_embeds = adynorm.get_embeddings(mentions=val_dictionary_names, use_tqdm=use_tqdm)

        val_score_matrix = adynorm.get_score_matrix(
            mention_embeddings=val_concepts_embeds,
            dictionary_embeddings=val_dictionary_embeds
        )
        val_candidate_idxs = adynorm.get_candidate_indexes(
            score_matrix=val_score_matrix,
            k=k
        )
        val_dataset.load_dense_idxs(val_candidate_idxs)

        loss = val(
            model=model,
            loader=val_loader,
            criterion=criterion,
            beta=beta
        )

        losses['val'].append(loss)
        if scheduler:
            scheduler.step(loss)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'adynorm_ch_{epoch}_lr_sum.ckpt')
            adynorm.save(f'adynorm_ch_{epoch}_lr_sum')

    return model, optimizer, losses
