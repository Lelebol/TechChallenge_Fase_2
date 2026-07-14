# Tech Challenge – Fase 2
## Classificação da qualidade de vinhos (Wine Quality Dataset)

Pós-graduação FIAP/POSTECH

Modelo de **classificação binária** que prevê se um vinho é de **alta qualidade**
(nota sensorial ≥ 7) ou de **baixa/média qualidade** (nota < 7), a partir de 11 variáveis
físico-químicas de laboratório. O objetivo de negócio é apoiar o **controle de qualidade**,
sinalizando quais amostras têm maior probabilidade de receber nota alta.

---

## Problema

- **Tipo:** classificação binária supervisionada.
- **Alvo:** `high_quality = 1` se `quality ≥ 7`, senão `0` (a coluna original `quality` é preservada).
- **Desafio central:** forte **desbalanceamento de classes** (~14% de vinhos de alta qualidade,
  razão ~6:1), que orienta todas as decisões de validação, balanceamento e métricas.

## Dataset

- **Fonte:** Wine Quality Dataset (vinho tinto) – arquivo `data/WineQT.csv`.
- **Dimensões:** 1.143 amostras × 11 variáveis físico-químicas + `quality`.
- **Variáveis:** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides,
  free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol.
- **Qualidade dos dados:** 0 valores ausentes; 125 linhas exatamente duplicadas (removidas).

## Estrutura do repositório

```
TechChallenge_Fase_2/
├── data/
│   └── WineQT.csv                      # dataset
├── notebooks/
│   └── analise_wine_quality.ipynb      # análise completa (6 etapas), já executado
├── src/                                # funções reutilizáveis
│   ├── preprocessing.py                # carga, duplicatas, outliers, features, split, scaling
│   ├── eda.py                          # gráficos exploratórios
│   ├── modeling.py                     # modelos, desbalanceamento, GridSearchCV, SMOTE
│   └── evaluation.py                   # métricas, curvas, importância, ajuste de limiar
├── scripts/
│   └── run_notebook.py                 # executa o notebook end-to-end (ver nota de ambiente)
├── results/                            # saídas geradas (11 PNGs + CSV + JSON)
├── requirements.txt
└── README.md
```

## Como executar

### 1. Pré-requisitos
- Python 3.11+ (desenvolvido e testado em Python 3.14).

### 2. Instalação
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Rodar a análise

**Opção A – executar o notebook inteiro via script (recomendado neste ambiente):**
```bash
python scripts/run_notebook.py notebooks/analise_wine_quality.ipynb
```
O script executa todas as células, embute as saídas (tabelas e figuras) e regrava o `.ipynb`.

**Opção B – abrir interativamente no Jupyter:**
```bash
jupyter lab notebooks/analise_wine_quality.ipynb
```

> **Nota de ambiente.** Em Windows + Python 3.14, o kernel Jupyter padrão sobe mas o handshake
> TCP/zmq entre kernel e cliente não completa (erro *"Kernel didn't respond"*), o que quebra a
> execução via Jupyter/nbconvert. Por isso o `scripts/run_notebook.py` usa o **kernel in-process**
> do `ipykernel` (mesmo processo, canais em memória, sem TCP/zmq) — um shell Jupyter completo que
> produz tabelas e figuras normalmente. Em ambientes onde o Jupyter funciona, a Opção B também vale.

## Pipeline (6 etapas)

1. **Compreensão do problema** – carga, estrutura dos dados, criação do alvo binário.
2. **Análise exploratória (EDA)** – histogramas, correlações, boxplots, balanceamento de classes.
3. **Pré-processamento** – remoção de duplicatas, *capping* de outliers (IQR), 3 variáveis
   derivadas, split estratificado 80/20 e padronização — tudo **sem vazamento** (ajustado só no treino).
4. **Modelagem** – 3 modelos (Regressão Logística, Random Forest, XGBoost), tratamento de
   desbalanceamento (`class_weight`/`scale_pos_weight`, justificado vs. SMOTE) e **GridSearchCV** no RF.
5. **Avaliação** – accuracy, precision, recall, F1, ROC-AUC, PR-AUC; matrizes de confusão,
   curvas ROC e Precision-Recall; tabela comparativa; ajuste do limiar de decisão.
6. **Interpretação** – importância das variáveis (árvores) e coeficientes (linear), com discussão enológica.

## Resultados

Métricas no **conjunto de teste** (204 amostras, 27 positivas), ordenado por F1:

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Random Forest** | **0,902** | 0,630 | 0,630 | **0,630** | 0,909 | 0,685 |
| **Random Forest (ajustado)** | 0,882 | 0,543 | 0,704 | 0,613 | **0,913** | **0,710** |
| XGBoost | 0,878 | 0,542 | 0,481 | 0,510 | 0,897 | 0,616 |
| Regressão Logística | 0,799 | 0,375 | 0,778 | 0,506 | 0,884 | 0,469 |

- **Melhor equilíbrio (F1/accuracy):** Random Forest.
- **Melhor ranqueamento (ROC-AUC/PR-AUC):** Random Forest ajustado.
- **Maior recall:** Regressão Logística (0,778), ao custo de precisão.

Artefatos em `results/`: 5 gráficos de EDA, 3 de avaliação, 3 de interpretação,
`metricas_modelos.csv` e `metricas_modelos.json`.

## Principais insights

- **Variáveis que mais elevam a qualidade:** teor de **álcool** (o maior direcionador),
  **sulfatos** e **ácido cítrico**.
- **Variável que mais reduz a qualidade:** **acidez volátil** (ácido acético, aroma avinagrado) —
  o coeficiente negativo mais forte e a 3ª variável mais importante nas árvores.
- **Engenharia de atributos validada:** 2 das 3 variáveis derivadas aparecem no **top 4 de ambas
  as árvores** — `alcohol_sulphates` (interação corpo × conservante) e `fixed_volatile_ratio`
  (acidez "boa" vs. "ruim").
- Achados **coerentes com a EDA** (correlações) e com o conhecimento de **enologia**.

## Decisões metodológicas

- **Sem vazamento de dados:** *capping* de outliers, padronização e escolha de limiar são
  ajustados **apenas no treino**; duplicatas removidas **antes** do split.
- **Desbalanceamento:** aprendizado sensível ao custo (`class_weight='balanced'` /
  `scale_pos_weight`), escolhido após comparação com **SMOTE** por validação cruzada (empate técnico,
  mas sem gerar dados sintéticos).
- **Métricas adequadas:** priorizam-se F1, ROC-AUC e **PR-AUC** (mais honesta sob desbalanceamento)
  em vez da acurácia, que é enganosa quando 87% das amostras são da classe majoritária.
- **Ajuste de limiar** por probabilidades *out-of-fold* (sem tocar no teste), demonstrando o
  trade-off recall × precisão como alavanca de negócio.
