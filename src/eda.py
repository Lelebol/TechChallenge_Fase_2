"""Funcoes de Analise Exploratoria de Dados (EDA).

Cada funcao gera um grafico, salva um PNG em `results/` e devolve um objeto util
(caminho salvo, matriz de correlacao ou contagem de classes) para uso na narrativa do notebook.
Os graficos sao exibidos automaticamente inline no Jupyter (backend inline).
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_plot_style() -> None:
    """Define um estilo visual consistente para todos os graficos."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 120
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["figure.autolayout"] = False


def _grid_shape(n: int, ncols: int = 4) -> tuple[int, int]:
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def plot_feature_histograms(
    df: pd.DataFrame, cols: list[str], results_dir: str | Path, filename: str = "eda_histogramas.png"
) -> Path:
    """Histograma (com KDE) de cada variavel numerica, em grade."""
    nrows, ncols = _grid_shape(len(cols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="#4C72B0")
        axes[i].set_title(col)
        axes[i].set_xlabel("")
    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Distribuicao das variaveis numericas", fontsize=14, y=1.02)
    fig.tight_layout()
    path = Path(results_dir) / filename
    fig.savefig(path, bbox_inches="tight")
    return path


def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: list[str],
    results_dir: str | Path,
    filename: str = "eda_correlacao_heatmap.png",
) -> pd.DataFrame:
    """Matriz de correlacao (Pearson) com heatmap. Retorna a matriz para inspecao no notebook."""
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(1.05 * len(cols), 0.9 * len(cols)))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Matriz de correlacao", fontsize=14)
    fig.tight_layout()
    path = Path(results_dir) / filename
    fig.savefig(path, bbox_inches="tight")
    return corr


def plot_boxplots(
    df: pd.DataFrame, cols: list[str], results_dir: str | Path, filename: str = "eda_boxplots.png"
) -> Path:
    """Boxplot de cada variavel numerica para inspecao visual de outliers."""
    nrows, ncols = _grid_shape(len(cols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.boxplot(y=df[col], ax=axes[i], color="#55A868")
        axes[i].set_title(col)
        axes[i].set_ylabel("")
    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Boxplots das variaveis numericas (deteccao de outliers)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = Path(results_dir) / filename
    fig.savefig(path, bbox_inches="tight")
    return path


def plot_boxplots_by_target(
    df: pd.DataFrame,
    cols: list[str],
    target: str,
    results_dir: str | Path,
    filename: str = "eda_boxplots_por_classe.png",
) -> Path:
    """Boxplots de cada variavel separados pela classe alvo.

    Ajuda a ver quais variaveis mais separam vinhos de alta vs baixa/media qualidade.
    """
    nrows, ncols = _grid_shape(len(cols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.boxplot(data=df, x=target, y=col, ax=axes[i], hue=target, palette="Set2", legend=False)
        axes[i].set_title(col)
        axes[i].set_xlabel("high_quality")
        axes[i].set_ylabel("")
    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Variaveis por classe (0 = baixa/media, 1 = alta qualidade)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = Path(results_dir) / filename
    fig.savefig(path, bbox_inches="tight")
    return path


def plot_class_balance(
    df: pd.DataFrame,
    target: str,
    results_dir: str | Path,
    filename: str = "eda_balanceamento_classes.png",
) -> pd.Series:
    """Grafico de barras do balanceamento das classes. Retorna a contagem por classe."""
    counts = df[target].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, hue=counts.index.astype(str),
                palette="Set1", legend=False, ax=ax)
    total = counts.sum()
    for i, v in enumerate(counts.values):
        ax.text(i, v + total * 0.01, f"{v}\n({v / total:.1%})", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("high_quality (0 = baixa/media, 1 = alta)")
    ax.set_ylabel("Quantidade de vinhos")
    ax.set_title("Balanceamento das classes")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    path = Path(results_dir) / filename
    fig.savefig(path, bbox_inches="tight")
    return counts
