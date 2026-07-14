"""Definicao dos modelos, tratamento de desbalanceamento e busca de hiperparametros."""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

RANDOM_STATE = 42


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Razao negativos/positivos, usada no XGBoost para compensar o desbalanceamento."""
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    return n_neg / max(n_pos, 1)


def get_models(random_state: int = RANDOM_STATE, scale_pos_weight: float = 1.0) -> dict:
    """Retorna os modelos base, ja configurados para lidar com o desbalanceamento.

    Escolha dos 3 modelos (diversidade de familias/algoritmos):
      - Regressao Logistica: baseline linear e altamente interpretavel (coeficientes = efeito
        de cada variavel). Sensivel a escala -> usamos dados padronizados.
      - Random Forest: ensemble de arvores (bagging), captura relacoes nao lineares e
        interacoes, robusto a outliers e fornece importancia por impureza.
      - XGBoost: gradient boosting, referencia em dados tabulares, costuma entregar o melhor
        desempenho preditivo.

    Estrategia de desbalanceamento: aprendizado sensivel ao custo -> class_weight='balanced'
    (LogReg e Random Forest) e scale_pos_weight (XGBoost). Nao cria dados sinteticos e reforca
    o peso da classe minoritaria diretamente na funcao de perda.
    """
    return {
        "Regressao Logistica": LogisticRegression(
            class_weight="balanced", max_iter=5000, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def fit_models(models: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Treina (fit) cada modelo do dicionario e retorna um novo dicionario com os modelos ajustados."""
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


# Grade de hiperparametros para o Random Forest (busca via GridSearchCV).
RF_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 8, 16],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}


def tune_model(
    estimator,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "roc_auc",
    cv: int = 5,
    random_state: int = RANDOM_STATE,
) -> GridSearchCV:
    """Busca de hiperparametros com GridSearchCV e validacao cruzada estratificada.

    Retorna o objeto GridSearchCV ja ajustado (best_estimator_, best_params_, cv_results_).
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=skf,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search


def make_smote_pipeline(estimator, random_state: int = RANDOM_STATE) -> ImbPipeline:
    """Pipeline com SMOTE + estimador, para comparar oversampling vs. class_weight.

    O SMOTE fica dentro do pipeline para ser aplicado apenas no treino de cada fold da CV,
    evitando vazamento (nunca sintetiza amostras no conjunto de validacao/teste).
    """
    return ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=random_state)),
            ("clf", estimator),
        ]
    )


def compare_imbalance_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "f1",
    cv: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compara tres estrategias de desbalanceamento via validacao cruzada estratificada.

    Usa a Regressao Logistica como estimador base (rapida e sensivel ao balanceamento), variando
    apenas o tratamento do desbalanceamento para uma comparacao justa:
      - Sem tratamento (baseline);
      - class_weight='balanced' (custo sensivel na funcao de perda);
      - SMOTE (oversampling sintetico da classe minoritaria, aplicado so no treino de cada fold).

    Retorna um DataFrame com media e desvio-padrao do score em cada estrategia.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    base = LogisticRegression(max_iter=5000, random_state=random_state)
    s_base = cross_val_score(base, X_train, y_train, scoring=scoring, cv=skf, n_jobs=-1)

    weighted = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=random_state)
    s_weighted = cross_val_score(weighted, X_train, y_train, scoring=scoring, cv=skf, n_jobs=-1)

    smote_pipe = make_smote_pipeline(
        LogisticRegression(max_iter=5000, random_state=random_state), random_state=random_state
    )
    s_smote = cross_val_score(smote_pipe, X_train, y_train, scoring=scoring, cv=skf, n_jobs=-1)

    return pd.DataFrame(
        {
            f"{scoring}_medio": [s_base.mean(), s_weighted.mean(), s_smote.mean()],
            f"{scoring}_desvio": [s_base.std(), s_weighted.std(), s_smote.std()],
        },
        index=["Sem tratamento", "class_weight='balanced'", "SMOTE"],
    )
