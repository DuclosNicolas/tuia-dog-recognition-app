"""Funciones auxiliares para evaluacion (provistas por la catedra).

Cubren las metricas pedidas en el TP:
  - Etapa 1: NDCG@10 (ndcg_at_k).
  - Etapa 2: precision / recall / F1 / specificity.
"""
from __future__ import annotations

import math
from typing import Sequence


def ndcg_at_k(
    relevances: Sequence[float], k: int = 10, n_relevant: int | None = None
) -> float:
    """NDCG@k de un ranking con relevancia binaria.

    `relevances` son las relevancias en el orden devuelto por la busqueda
    (1.0 si el vecino es de la misma raza que la consulta, 0.0 si no).

    `n_relevant` es la cantidad de imagenes relevantes en toda la base (ej:
    imagenes de la raza de la consulta en la galeria). El ranking ideal ubica
    min(n_relevant, k) imagenes relevantes en las primeras posiciones; si se
    omite, se asume que la base tiene al menos k. El ideal no se calcula
    reordenando los resultados devueltos: asi, un ranking con un unico acierto
    en la primera posicion no obtiene NDCG = 1.
    """
    rel = [float(r) for r in list(relevances)[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    n_ideal = k if n_relevant is None else min(int(n_relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    return dcg / idcg if idcg > 0 else 0.0


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Retorna (precision, recall, f1) a partir de conteos TP/FP/FN."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def specificity(tn: int, fp: int) -> float:
    """Especificidad: TN / (TN + FP)."""
    return tn / (tn + fp) if (tn + fp) else 0.0
