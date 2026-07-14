"""Funcoes de carregamento e pre-processamento dos dados de vinho.

Todas as funcoes sao puras (nao alteram o DataFrame de entrada) para facilitar o uso
dentro do notebook sem efeitos colaterais inesperados.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Colunas fisico-quimicas originais do dataset (na ordem em que aparecem no CSV).
ORIGINAL_FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

TARGET_ORIGINAL = "quality"
TARGET_BINARY = "high_quality"


def load_data(csv_path: str) -> pd.DataFrame:
    """Le o CSV e remove a coluna 'Id' (apenas um identificador, sem valor preditivo)."""
    df = pd.read_csv(csv_path)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    return df


def create_binary_target(df: pd.DataFrame, threshold: int = 7) -> pd.DataFrame:
    """Cria a variavel alvo binaria 'high_quality' (1 se quality >= threshold, senao 0).

    A coluna original 'quality' e preservada.
    """
    out = df.copy()
    out[TARGET_BINARY] = (out[TARGET_ORIGINAL] >= threshold).astype(int)
    return out


def remove_exact_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    """Remove linhas exatamente duplicadas (mantendo a primeira ocorrencia).

    Justificativa: registros identicos que caiam em treino E teste inflam artificialmente
    a metrica de teste (vazamento). Retorna o DataFrame limpo e a quantidade removida.
    """
    before = len(df)
    out = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    return out, before - len(out)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas com justificativa de dominio (enologia).

    - free_sulfur_ratio: fracao do SO2 que esta livre (free / total). Apenas o SO2 livre
      protege o vinho contra oxidacao e microrganismos; um valor baixo indica que a maior
      parte do SO2 esta ligada e ja nao e eficaz.
    - fixed_volatile_ratio: equilibrio entre acidez fixa ("boa", tartarica) e acidez volatil
      ("ruim", acetica/avinagrada). Razoes altas tendem a indicar vinhos mais frescos e menos
      deteriorados.
    - alcohol_sulphates: interacao entre dois dos maiores impulsionadores positivos da
      qualidade (corpo alcoolico e sulfatos, que atuam como conservante/antimicrobiano).
    """
    out = df.copy()
    eps = 1e-6  # evita divisao por zero
    out["free_sulfur_ratio"] = out["free sulfur dioxide"] / (out["total sulfur dioxide"] + eps)
    out["fixed_volatile_ratio"] = out["fixed acidity"] / (out["volatile acidity"] + eps)
    out["alcohol_sulphates"] = out["alcohol"] * out["sulphates"]
    return out


ENGINEERED_FEATURES = ["free_sulfur_ratio", "fixed_volatile_ratio", "alcohol_sulphates"]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Lista as colunas de features (fisico-quimicas + derivadas), excluindo alvos."""
    exclude = {TARGET_ORIGINAL, TARGET_BINARY}
    return [c for c in df.columns if c not in exclude]


def fit_outlier_caps(df: pd.DataFrame, cols: list[str], k: float = 1.5) -> dict[str, tuple[float, float]]:
    """Calcula os limites de capping por IQR (Q1 - k*IQR, Q3 + k*IQR) para cada coluna.

    Ajustado apenas no conjunto de TREINO para evitar vazamento de informacao.
    """
    bounds: dict[str, tuple[float, float]] = {}
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - k * iqr, q3 + k * iqr)
    return bounds


def apply_outlier_caps(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Aplica capping (winsorizacao) usando os limites informados, preservando todas as linhas."""
    out = df.copy()
    for col, (low, high) in bounds.items():
        out[col] = out[col].clip(lower=low, upper=high)
    return out


def count_outliers(df: pd.DataFrame, cols: list[str], k: float = 1.5) -> pd.Series:
    """Conta quantos valores caem fora dos limites IQR em cada coluna (para relatorio de EDA)."""
    counts = {}
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        counts[col] = int(((df[col] < low) | (df[col] > high)).sum())
    return pd.Series(counts).sort_values(ascending=False)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Divisao treino/teste estratificada pela variavel alvo (por causa do desbalanceamento)."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Padroniza as features com StandardScaler (ajustado somente no treino).

    Retorna (X_train_scaled, X_test_scaled, scaler) mantendo os DataFrames com nomes de coluna,
    o que e util para a interpretacao de coeficientes/importancias depois.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler
