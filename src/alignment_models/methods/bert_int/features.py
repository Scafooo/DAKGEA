"""Feature extraction for the interaction model stage."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.alignment_models.methods.bert_int.dual_aggregation import (
    dual_aggregation_features,
    kernel_mus,
    kernel_sigmas,
)
from src.alignment_models.methods.bert_int.similarity import cosine_similarity_matrix
from src.logger import get_logger

logger = get_logger(__name__)


def build_neighbor_dict(
    rel_triples: Sequence[Tuple[int, int, int]],
    entities: Sequence[int],
    max_neighbors: int,
    pad_id: int,
) -> Dict[int, List[int]]:
    neigh: Dict[int, List[int]] = {ent: [] for ent in entities}
    for head, _, tail in rel_triples:
        neigh.setdefault(head, []).append(tail)
        neigh.setdefault(tail, []).append(head)
    for entity, values in neigh.items():
        np.random.shuffle(values)
        neigh[entity] = values[:max_neighbors]
    for entity in neigh:
        padded = neigh[entity] + [pad_id] * (max_neighbors - len(neigh[entity]))
        neigh[entity] = padded
    return neigh


def build_attribute_values(
    attribute_triples: Sequence[Tuple[int, str, str, str]],
    entities: Sequence[int],
    max_values: int,
    fallback_names: Dict[int, str],
    pad_token: str,
) -> Dict[int, List[str]]:
    ent2values: Dict[int, List[str]] = {ent: [] for ent in entities}
    for ent, attr, val, _ in attribute_triples:
        ent2values.setdefault(ent, []).append(val)
    for ent in entities:
        if ent in fallback_names:
            name_val = fallback_names[ent]
            if name_val and name_val not in ent2values[ent]:
                ent2values[ent].append(name_val)
        np.random.shuffle(ent2values[ent])
        values = ent2values[ent][:max_values]
        values += [pad_token] * (max_values - len(values))
        ent2values[ent] = values
    return ent2values


def clean_attribute_triples(
    attribute_triples: Sequence[Tuple[int, str, str, str]],
    threshold: int = 3,
) -> List[Tuple[int, str, str, str]]:
    counts: Dict[Tuple[int, str], int] = {}
    for ent, attr, _, _ in attribute_triples:
        counts[(ent, attr)] = counts.get((ent, attr), 0) + 1

    cleaned: List[Tuple[int, str, str, str]] = []
    for triple in attribute_triples:
        ent, attr, _, _ = triple
        if counts.get((ent, attr), 0) <= threshold:
            cleaned.append(triple)
    return cleaned


def neighbor_features(
    entity_pairs: Sequence[Tuple[int, int]],
    entity_embeddings: torch.Tensor,
    neighbor_dict: Dict[int, List[int]],
    pad_id: int,
    kernel_num: int,
    device: torch.device,
    batch_size: int = 512,
) -> List[List[float]]:
    mus = kernel_mus(kernel_num).to(device).view(1, 1, -1)
    sigmas = kernel_sigmas(kernel_num).to(device).view(1, 1, -1)

    features: List[List[float]] = []
    for start in range(0, len(entity_pairs), batch_size):
        batch_pairs = entity_pairs[start : start + batch_size]
        e1s = [pair[0] for pair in batch_pairs]
        e2s = [pair[1] for pair in batch_pairs]
        neigh1 = torch.tensor([neighbor_dict[e] for e in e1s], dtype=torch.long, device=device)
        neigh2 = torch.tensor([neighbor_dict[e] for e in e2s], dtype=torch.long, device=device)

        mask1 = (neigh1 != pad_id).float().unsqueeze(-1)
        mask2 = (neigh2 != pad_id).float().unsqueeze(-1)

        emb1 = entity_embeddings[neigh1]
        emb2 = entity_embeddings[neigh2]
        sim_matrix = torch.bmm(emb1, emb2.transpose(1, 2))
        feats = dual_aggregation_features(sim_matrix, mus, sigmas, mask1, mask2)
        features.extend(feats.detach().cpu().tolist())
    logger.debug("[BERT-INT] Neighbor features computed for %d pairs.", len(features))
    return features


def description_features(
    entity_pairs: Sequence[Tuple[int, int]],
    entity_embeddings: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> List[List[float]]:
    features: List[List[float]] = []
    for start in range(0, len(entity_pairs), batch_size):
        batch_pairs = entity_pairs[start : start + batch_size]
        e1s = torch.tensor([pair[0] for pair in batch_pairs], dtype=torch.long, device=device)
        e2s = torch.tensor([pair[1] for pair in batch_pairs], dtype=torch.long, device=device)
        emb1 = entity_embeddings[e1s]
        emb2 = entity_embeddings[e2s]
        sim = F.cosine_similarity(emb1, emb2).unsqueeze(-1)
        # Convert to list of lists: each row is [similarity_score]
        features.extend(sim.detach().cpu().tolist())
    logger.debug("[BERT-INT] Description features computed for %d pairs.", len(features))
    return features


def attribute_value_embeddings(
    model: torch.nn.Module,
    values: Sequence[str],
    tokenizer,
    batch_size: int,
    device: torch.device,
    max_length: int = 64,
) -> Tuple[List[List[float]], List[str]]:
    embeddings: List[List[float]] = []
    model.eval()
    for start in range(0, len(values), batch_size):
        batch_vals = list(values[start : start + batch_size])
        encoded = tokenizer(
            batch_vals,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = encoded["input_ids"].to(device)
        masks = encoded["attention_mask"].to(device)
        with torch.no_grad():
            emb = model(tokens, masks)
        embeddings.extend(emb.detach().cpu().tolist())
    logger.debug("[BERT-INT] Encoded %d attribute values.", len(embeddings))
    return embeddings, list(values)


def attribute_features(
    entity_pairs: Sequence[Tuple[int, int]],
    value_embeddings: torch.Tensor,
    ent2value_ids: Dict[int, List[int]],
    pad_id: int,
    kernel_num: int,
    device: torch.device,
    batch_size: int = 512,
) -> List[List[float]]:
    mus = kernel_mus(kernel_num).to(device).view(1, 1, -1)
    sigmas = kernel_sigmas(kernel_num).to(device).view(1, 1, -1)

    features: List[List[float]] = []
    for start in range(0, len(entity_pairs), batch_size):
        batch_pairs = entity_pairs[start : start + batch_size]
        e1s = [pair[0] for pair in batch_pairs]
        e2s = [pair[1] for pair in batch_pairs]
        values1 = torch.tensor([ent2value_ids[e] for e in e1s], dtype=torch.long, device=device)
        values2 = torch.tensor([ent2value_ids[e] for e in e2s], dtype=torch.long, device=device)

        mask1 = (values1 != pad_id).float().unsqueeze(-1)
        mask2 = (values2 != pad_id).float().unsqueeze(-1)

        emb1 = value_embeddings[values1]
        emb2 = value_embeddings[values2]
        sim_matrix = torch.bmm(emb1, emb2.transpose(1, 2))
        feats = dual_aggregation_features(sim_matrix, mus, sigmas, mask1, mask2)
        features.extend(feats.detach().cpu().tolist())
    logger.debug("[BERT-INT] Attribute features computed for %d pairs.", len(features))
    return features
