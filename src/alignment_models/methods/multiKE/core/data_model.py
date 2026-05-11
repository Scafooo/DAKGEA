"""DataModel: builds KGS + literal/name/value embeddings from a DAKGEA Dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from rdflib import Literal, URIRef
from sklearn import preprocessing

from .kg import KG
from .kgs import KGs
from .literal_encoder import (
    clear_attribute_triples,
    encode_literals,
    read_word2vec,
)

if TYPE_CHECKING:
    from src.core.dataset import Dataset


_LABEL_PREDICATES = [
    'http://xmlns.com/foaf/0.1/name',              # BBC, DBpedia
    'http://purl.org/dc/elements/1.1/title',        # BBC album titles
    'http://www.w3.org/2000/01/rdf-schema#label',   # DBpedia, general
    'http://www.w3.org/2004/02/skos/core#altLabel', # Wikidata (no rdfs:label in OpenEA)
    'http://www.w3.org/2000/01/rdf-schema#comment',
]

_LABEL_PRED_RANK = {p: i for i, p in enumerate(_LABEL_PREDICATES)}


def _local_name(uri: str) -> str:
    if '#' in uri:
        name = uri.split('#')[-1]
    else:
        name = uri.split('/')[-1]
    return name.replace('_', ' ')


def _entity_labels_from_attr(attr_triples) -> Dict[str, str]:
    """Extract best name label for each entity from attribute triples.

    Priority order: foaf:name > dc:title > rdfs:label > rdfs:comment.
    Falls back to URI local name for entities without any label predicate.
    """
    best: Dict[str, Tuple[int, str]] = {}
    for e, a, v in attr_triples:
        rank = _LABEL_PRED_RANK.get(a)
        if rank is None:
            continue
        if e not in best or rank < best[e][0]:
            best[e] = (rank, v)
    return {e: label for e, (_, label) in best.items()}


def _extract_triples(rdf_graph):
    """Split an rdflib Graph into relation triples (URIRef obj) and raw attr triples (Literal obj)."""
    rel_triples = set()
    attr_triples = set()
    for s, p, o in rdf_graph:
        s_str, p_str = str(s), str(p)
        if isinstance(o, Literal):
            attr_triples.add((s_str, p_str, str(o)))
        else:
            rel_triples.add((s_str, p_str, str(o)))
    return rel_triples, attr_triples


class DataModel:
    """
    Builds the MultiKE data model from a DAKGEA Dataset in memory.

    Parameters
    ----------
    dataset : Dataset
        DAKGEA Dataset with knowledge_graph_source / knowledge_graph_target / aligned_entities.
    word2vec_path : str or None
        Path to a pre-trained word2vec text file.  If None, literal embeddings are random.
    dim : int
        Output dimension for literal embeddings (and all model embeddings).
    encoder_epochs : int
        Number of AutoEncoder training epochs.
    encoder_active : str
        Activation for the AutoEncoder ('tanh' or 'sigmoid').
    encoder_normalize : bool
        Whether to L2-normalise AutoEncoder output.
    literal_normalize : bool
        Whether to L2-normalise the final literal/name/value embedding matrices.
    optimizer : str
        Optimizer name for the AutoEncoder ('Adagrad', 'Adam', etc.).
    train_ratio / valid_ratio : float
        Fractions of alignment pairs used for train / valid splits (rest goes to test).
    mode : str
        KGs construction mode ('swapping' preserves cross-KG swapping, 'mapping' is simpler).
    """

    def __init__(
        self,
        dataset: "Dataset",
        word2vec_path: Optional[str] = None,
        dim: int = 75,
        encoder_epochs: int = 100,
        encoder_active: str = 'tanh',
        encoder_normalize: bool = True,
        literal_normalize: bool = True,
        optimizer: str = 'Adagrad',
        learning_rate: float = 0.001,
        batch_size: int = 512,
        train_ratio: float = 0.2,
        valid_ratio: float = 0.1,
        mode: str = 'swapping',
    ):
        self.dim = dim
        self.literal_normalize = literal_normalize

        # --- Build raw URI-level KG objects ---
        rel_triples1, raw_attr1 = _extract_triples(dataset.knowledge_graph_source)
        rel_triples2, raw_attr2 = _extract_triples(dataset.knowledge_graph_target)

        # Clean attribute triples BEFORE KGs so attribute predicates get integer IDs assigned
        cleaned1 = clear_attribute_triples(raw_attr1)
        cleaned2 = clear_attribute_triples(raw_attr2)

        uri_kg1 = KG(rel_triples1, cleaned1)
        uri_kg2 = KG(rel_triples2, cleaned2)

        # --- Split alignment pairs ---
        all_pairs = sorted((str(e1), str(e2)) for e1, e2 in dataset.aligned_entities)
        n = len(all_pairs)
        n_train = max(1, int(n * train_ratio))
        n_valid = max(1, int(n * valid_ratio))
        train_links = all_pairs[:n_train]
        valid_links = all_pairs[n_train:n_train + n_valid]

        fixed_test = getattr(dataset, 'fixed_test_pairs', None)
        if fixed_test:
            test_links = sorted((str(e1), str(e2)) for e1, e2 in fixed_test)
            print(f'[MultiKE] splits: train={len(train_links)}, valid={len(valid_links)}, '
                  f'test={len(test_links)} (fixed from forget_labels)')
        else:
            test_links = all_pairs[n_train + n_valid:]
            print(f'[MultiKE] splits: train={len(train_links)}, valid={len(valid_links)}, '
                  f'test={len(test_links)} (internal split)')

        # --- Ensure every aligned entity has at least one relation triple ---
        # PLM-generated synthetic entities may have only attribute triples; KG.entities_set
        # is built exclusively from relation triple heads/tails, so without a relation triple
        # these entities get no integer ID and cause a KeyError in uris_pair_2ids.
        # Fix: inject a self-loop stub triple for any entity missing from the relation graph.
        _STUB_REL = 'http://dakgea.local/synthetic_entity'
        stub1, stub2 = 0, 0
        for e1, e2 in train_links + valid_links + test_links:
            if e1 not in uri_kg1.entities_set:
                uri_kg1.relation_triples_set.add((e1, _STUB_REL, e1))
                stub1 += 1
            if e2 not in uri_kg2.entities_set:
                uri_kg2.relation_triples_set.add((e2, _STUB_REL, e2))
                stub2 += 1
        if stub1 or stub2:
            uri_kg1.set_relations(uri_kg1.relation_triples_set)
            uri_kg2.set_relations(uri_kg2.relation_triples_set)
            print(f'[MultiKE] injected stub relation triples for {stub1} KG1 + {stub2} KG2 '
                  f'synthetic entities (attribute-only, no relation triples)')

        # --- Build KGS (assigns integer IDs, including for attribute predicates) ---
        self.kgs = KGs(uri_kg1, uri_kg2, train_links, valid_links, test_links=test_links, mode=mode)

        # Entity local names (use attribute labels when available, fallback to URI local name)
        entities_uri = set(self.kgs.uri_kg1.entities_set) | set(self.kgs.uri_kg2.entities_set)
        attr_labels = _entity_labels_from_attr(list(raw_attr1) + list(raw_attr2))
        local_name_dict: Dict[str, str] = {
            e: (attr_labels.get(e) or _local_name(e)) for e in entities_uri
        }
        n_labeled = sum(1 for e in entities_uri if e in attr_labels)
        print(f'[MultiKE] entity labels from attributes: {n_labeled}/{len(entities_uri)} '
              f'(fallback to URI local name for the rest)')
        name_list = list(local_name_dict.values())

        # All value strings
        value_strings = list({v for _, _, v in cleaned1 + cleaned2})
        all_literals = list(set(value_strings + name_list))
        print(f'[MultiKE] literals: names={len(name_list)}, values={len(value_strings)}, total={len(all_literals)}')

        literal_embeddings = self._encode_literals(
            all_literals, word2vec_path, dim, encoder_epochs, encoder_active,
            encoder_normalize, optimizer, learning_rate, batch_size)

        literal_id_dic = {lit: i for i, lit in enumerate(all_literals)}

        # --- Name embedding matrix (one row per entity in ID order) ---
        n_ents = self.kgs.entities_num
        id_to_uri1 = {v: k for k, v in self.kgs.kg1.entities_id_dict.items()}
        id_to_uri2 = {v: k for k, v in self.kgs.kg2.entities_id_dict.items()}
        id_to_uri = {**id_to_uri1, **id_to_uri2}
        name_indices = []
        for i in range(n_ents):
            uri = id_to_uri.get(i, '')
            name = local_name_dict.get(uri, '')
            name_indices.append(literal_id_dic.get(name, 0))
        name_mat = literal_embeddings[name_indices, :]
        if literal_normalize:
            name_mat = preprocessing.normalize(name_mat)
        self.local_name_vectors = name_mat.astype(np.float32)

        # --- Attribute value vectors ---
        val_list = list({v for _, _, v in cleaned1 + cleaned2})
        val_id = {v: i for i, v in enumerate(val_list)}

        # Build int-ID attr triples: (int_entity_id, int_attr_id, int_val_id)
        eid1 = self.kgs.kg1.entities_id_dict
        aid1 = self.kgs.kg1.attributes_id_dict   # now populated (KG built with cleaned triples)
        eid2 = self.kgs.kg2.entities_id_dict
        aid2 = self.kgs.kg2.attributes_id_dict

        id_attr1 = set()
        for e, a, v in cleaned1:
            if e in eid1 and a in aid1:
                id_attr1.add((eid1[e], aid1[a], val_id[v]))
        id_attr2 = set()
        for e, a, v in cleaned2:
            if e in eid2 and a in aid2:
                id_attr2.add((eid2[e], aid2[a], val_id[v]))

        self.kgs.kg1.set_attributes(id_attr1)
        self.kgs.kg2.set_attributes(id_attr2)

        # Cross-KG supervised attribute triples (swapping)
        from .read import generate_sup_attribute_triples
        sup1, sup2 = generate_sup_attribute_triples(
            self.kgs.train_links, self.kgs.kg1.av_dict, self.kgs.kg2.av_dict)
        self.kgs.kg1.add_sup_attribute_triples(sup1)
        self.kgs.kg2.add_sup_attribute_triples(sup2)

        # Value embedding matrix ordered by val_id
        val_lit_indices = [literal_id_dic.get(v, 0) for v in val_list]
        value_mat = literal_embeddings[val_lit_indices, :]
        if literal_normalize:
            value_mat = preprocessing.normalize(value_mat)
        self.value_vectors = value_mat.astype(np.float32)

        # Full literal embeddings matrix (indexed by literal_id_dic)
        if literal_normalize:
            literal_embeddings = preprocessing.normalize(literal_embeddings)
        self.literal_vectors = literal_embeddings.astype(np.float32)

        print(f'[MultiKE] name_mat: {self.local_name_vectors.shape}, value_mat: {self.value_vectors.shape}')

    @staticmethod
    def _encode_literals(literal_list, word2vec_path, dim, epochs, activation,
                          normalize, optimizer, learning_rate, batch_size):
        if not literal_list:
            return np.zeros((0, dim), dtype=np.float32)

        if word2vec_path:
            print(f'[MultiKE] loading word2vec from {word2vec_path}')
            word2vec = read_word2vec(word2vec_path)
        else:
            print('[MultiKE] no word2vec path; using random literal embeddings')
            rng = np.random.default_rng(42)
            return rng.standard_normal((len(literal_list), dim)).astype(np.float32)

        return encode_literals(
            literal_list, word2vec, dim=dim,
            activation=activation, normalize=normalize,
            epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, optimizer_name=optimizer)
