"""PLM-based augmentation strategy leveraging latent interpolation."""

from __future__ import annotations

from src.augmentation.base import AugmentationMethod
from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.core.dataset import Dataset


@AUGMENTATION_REGISTRY.register("plm_augmentation")
class PLMAugmenter(AugmentationMethod):
    """Augment datasets via BART-based latent interpolation of literal attributes."""

    registry_name = "plm_augmentation"

    def __init__(self, config):
        super().__init__(config)

    def augment(self, dataset: Dataset) -> Dataset:
        # TODO: Implement PLM-based augmentation

        # Prendi un nodo set random
        # Crea due nuovi nodi con due id uno coerente con il primo grafo e uno coerente con il seconod grafo
        # Espandi i due nodi con i rispetti attribuiti interpolazione etc.
        # Richiama Augmentation in maniera ricorsiva su nodo vicino se anche questo set/allineato
        # Altrimenti se non è un nodo set fai una semplice augmentation
        # Richiama nuovamente fino ad arrivare una profondità massima "d" data
        # Se il vicino di un nodo non set è un nodo set fera quelli da espandere allora devi espandere seguendo quanto fatto all`inizio
        # se non ci sono più vicini allora verifica che tutti quelli che dovevano essere espansi sono stati espansi in quanto il grafo potrebbe non essere connesso.


        return dataset
