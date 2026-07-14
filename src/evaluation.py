"""Avaliacao de modelos: metricas, matriz de confusao, curvas ROC/PR, tabela comparativa
e importancia das variaveis."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

CLASS_LABELS = ["Baixa/Media (0)", "Alta (1)"]


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Calcula as principais metricas para classificacao binaria desbalanceada."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def get_predictions(fitted_models: dict, X_test: pd.DataFrame) -> dict:
    """Devolve, para cada modelo, as classes previstas e a probabilidade da classe positiva."""
    preds = {}
    for name, model in fitted_models.items():
        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        preds[name] = {"pred": pred, "proba": proba}
    return preds


def evaluate_models(fitted_models: dict, X_test: pd.DataFrame, y_test: pd.Series):
    """Avalia todos os modelos e retorna (tabela_de_metricas, dicionario_de_predicoes)."""
    preds = get_predictions(fitted_models, X_test)
    rows = []
    for name, p in preds.items():
        m = compute_metrics(y_test, p["pred"], p["proba"])
        m["modelo"] = name
        rows.append(m)
    metrics_df = pd.DataFrame(rows).set_index("modelo")
    return metrics_df, preds


def build_comparison_table(metrics_df: pd.DataFrame, sort_by: str = "f1") -> pd.DataFrame:
    """Ordena e arredonda a tabela comparativa final."""
    ordered_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    cols = [c for c in ordered_cols if c in metrics_df.columns]
    return metrics_df[cols].sort_values(sort_by, ascending=False).round(4)


def save_metrics(metrics_df: pd.DataFrame, results_dir: str | Path,
                 csv_name: str = "metricas_modelos.csv", json_name: str = "metricas_modelos.json") -> tuple[Path, Path]:
    """Salva a tabela de metricas em CSV e JSON dentro de results/."""
    results_dir = Path(results_dir)
    csv_path = results_dir / csv_name
    json_path = results_dir / json_name
    metrics_df.round(4).to_csv(csv_path)
    payload = json.loads(metrics_df.round(4).reset_index().to_json(orient="records"))
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


def plot_confusion_matrices(preds: dict, y_test: pd.Series, results_dir: str | Path,
                            filename: str = "avaliacao_matrizes_confusao.png") -> Path:
    """Grade com a matriz de confusao de cada modelo."""
    names = list(preds.keys())
    ncols = len(names)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4))
    if ncols == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        cm = confusion_matrix(y_test, preds[name]["pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
    fig.suptitle("Matrizes de confusao (conjunto de teste)", fontsize=14, y=1.03)
    fig.tight_layout()
    path = Path(results_dir) / filename
    fig.savefig(path, bbox_inches="tight")
    return path


def plot_roc_curves(preds: dict, y_test: pd.Series, results_dir: str | Path,
                    filename: str = "avaliacao_curvas_roc.png") -> Path:
    """Curvas ROC de todos os modelos em um mesmo grafico, com AUC na legenda."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y_test, p["proba"])
        auc = roc_auc_score(y_test, p["proba"])
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Aleatorio (AUC = 0.500)")
    ax.set_xlabel("Taxa de falsos positivos (1 - especificidade)")
    ax.set_ylabel("Taxa de verdadeiros positivos (recall)")
    ax.set_title("Curva ROC - comparacao entre modelos")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = Path(results_dir) / filename
    fig.savefig(path, bbox_inches="tight")
    return path


def plot_pr_curves(preds: dict, y_test: pd.Series, results_dir: str | Path,
                   filename: str = "avaliacao_curvas_precision_recall.png") -> Path:
    """Curvas Precision-Recall (mais informativas sob desbalanceamento)."""
    baseline = float((y_test == 1).mean())
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, p in preds.items():
        prec, rec, _ = precision_recall_curve(y_test, p["proba"])
        ap = average_precision_score(y_test, p["proba"])
        ax.plot(rec, prec, linewidth=2, label=f"{name} (AP = {ap:.3f})")
    ax.axhline(baseline, color="k", linestyle="--", linewidth=1,
               label=f"Baseline (prevalencia = {baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall - comparacao entre modelos")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = Path(results_dir) / filename
    fig.savefig(path, bbox_inches="tight")
    return path


def get_feature_importance(model, feature_names: list[str]) -> pd.Series:
    """Extrai importancia das variaveis: feature_importances_ (arvores) ou coef_ (linear).

    Para modelos lineares retorna o coeficiente com sinal (interpretavel); para arvores,
    a importancia por impureza (sempre positiva).
    """
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = model.coef_.ravel()
    else:
        raise ValueError("Modelo nao expoe feature_importances_ nem coef_.")
    return pd.Series(values, index=feature_names)


def plot_feature_importance(importances: pd.Series, title: str, results_dir: str | Path,
                            filename: str, top_n: int | None = None, signed: bool = False) -> Path:
    """Grafico de barras horizontais da importancia das variaveis (ordenado por magnitude)."""
    s = importances.reindex(importances.abs().sort_values(ascending=True).index)
    if top_n is not None:
        s = s.tail(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(s))))
    if signed:
        colors = ["#C44E52" if v < 0 else "#4C72B0" for v in s.values]
        ax.barh(s.index, s.values, color=colors)
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_xlabel("Coeficiente (sinal indica direcao do efeito)")
    else:
        ax.barh(s.index, s.values, color="#4C72B0")
        ax.set_xlabel("Importancia")
    ax.set_title(title)
    fig.tight_layout()
    path = Path(results_dir) / filename
    fig.savefig(path, bbox_inches="tight")
    return path


def find_best_threshold(y_true, y_proba, metric: str = "f1") -> tuple[float, float]:
    """Varre limiares de decisao e retorna (melhor_threshold, melhor_valor) para a metrica dada."""
    thresholds = np.linspace(0.05, 0.95, 181)
    scorer = {"f1": f1_score, "recall": recall_score, "precision": precision_score}[metric]
    best_t, best_s = 0.5, -1.0
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        s = scorer(y_true, pred, zero_division=0)
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t), float(best_s)


def evaluate_at_threshold(y_true, y_proba, threshold: float) -> dict:
    """Recalcula metricas de ponto de operacao para um limiar especifico."""
    pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


def tune_threshold_cv(model, X_train, y_train, metric: str = "f1", cv: int = 5,
                      random_state: int = 42) -> tuple[float, float]:
    """Escolhe o limiar otimo a partir de probabilidades out-of-fold no treino.

    Gera probabilidades por validacao cruzada (cross_val_predict) - realistas, sem o otimismo de
    prever no proprio treino em que o modelo foi ajustado - e sem tocar no conjunto de teste.
    Retorna (melhor_limiar, melhor_valor_cv).
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    oof_proba = cross_val_predict(
        clone(model), X_train, y_train, cv=skf, method="predict_proba", n_jobs=-1
    )[:, 1]
    return find_best_threshold(y_train, oof_proba, metric=metric)
