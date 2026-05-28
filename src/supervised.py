from __future__ import annotations

import time
from pathlib import Path

import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
import matplotlib.pyplot as plt

from src.evaluation import evaluate_classifier
from src.utils import RANDOM_STATE


def _fit_grid(name: str, estimator, params: dict, X_train, y_train, X_val, y_val) -> tuple:
    """
    Treina um modelo usando GridSearchCV e avalia seu desempenho. O processo inclui a medição do tempo de treinamento, a busca pelos melhores hiperparâmetros com validação cruzada e a avaliação do modelo no conjunto de validação usando a métrica F1-score ponderada. O resultado é um dicionário contendo o nome do modelo, o melhor estimador encontrado, os melhores hiperparâmetros, a melhor pontuação de validação cruzada e a pontuação de validação, além do tempo gasto no treinamento.
    """
    start = time.perf_counter()
    grid = GridSearchCV(estimator, params, scoring="f1_weighted", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    validation_score = grid.score(X_val, y_val)
    return {
        "model_name": name,
        "model": grid.best_estimator_,
        "best_params": grid.best_params_,
        "cv_best_score": grid.best_score_,
        "validation_f1_weighted": validation_score,
        "train_time": train_time,
    }


def train_supervised_models(splits: dict, class_names: list[str], figures_dir: Path, models_dir: Path) -> list[dict]:
    """
    Treina e avalia modelos de classificação supervisionada, incluindo KNN (com e sem padronização), Árvore de Decisão e MLP/RNA. O processo envolve a definição de hiperparâmetros para cada modelo, a realização de busca em grade com validação cruzada para encontrar os melhores hiperparâmetros, a avaliação do desempenho no conjunto de validação usando F1-score ponderado e a avaliação final no conjunto de teste. Os resultados são salvos em arquivos e retornados em um dicionário.
    Args:        
        splits (dict): Um dicionário contendo os conjuntos de dados para treino, validação e teste.        class_names (list[str]): Uma lista de nomes das classes.        figures_dir (Path): O diretório onde as figuras serão salvas.        models_dir (Path): O diretório onde os modelos treinados serão salvos.
    Returns:        
        list[dict]: Uma lista de dicionários contendo os resultados de treinamento e avaliação de cada modelo de classificação supervisionada.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    results = []

    knn_scaled = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier())])
    knn_scaled_params = {
        "model__n_neighbors": [3, 5, 7, 9, 11, 15],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan"],
    }
    results.append(_fit_grid("KNN_padronizado", knn_scaled, knn_scaled_params, splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"]))

    knn_raw = KNeighborsClassifier()
    knn_raw_params = {"n_neighbors": [3, 5, 7, 9, 11, 15], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]}
    results.append(_fit_grid("KNN_sem_padronizacao", knn_raw, knn_raw_params, splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"]))

    tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    tree_params = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_leaf": [1, 3, 5, 10],
        "ccp_alpha": [0.0, 0.001, 0.005, 0.01],
    }
    results.append(_fit_grid("Arvore_Decisao", tree, tree_params, splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"]))

    mlp = Pipeline([("scaler", StandardScaler()), ("model", MLPClassifier(random_state=RANDOM_STATE, early_stopping=True, max_iter=700))])
    mlp_params = {
        "model__hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32)],
        "model__activation": ["relu", "tanh"],
        "model__learning_rate_init": [0.001, 0.01],
        "model__alpha": [0.0001, 0.001],
    }
    results.append(_fit_grid("MLP_RNA", mlp, mlp_params, splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"]))

    for result in results:
        result["test_metrics"] = evaluate_classifier(
            result["model"],
            splits["X_test"],
            splits["y_test"],
            class_names,
            result["model_name"],
            figures_dir,
        )
        joblib.dump(result["model"], models_dir / f"{result['model_name']}.joblib")

    tree_result = next(item for item in results if item["model_name"] == "Arvore_Decisao")
    tree_model = tree_result["model"]
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        tree_model,
        feature_names=splits["X"].columns.tolist(),
        class_names=class_names,
        filled=True,
        max_depth=3,
        fontsize=7,
        ax=ax,
    )
    ax.set_title("Árvore de Decisão - visualização limitada aos primeiros níveis")
    fig.tight_layout()
    fig.savefig(figures_dir / "decision_tree.png", dpi=150)
    plt.close(fig)
    export_graphviz(
        tree_model,
        out_file=str(models_dir / "decision_tree.dot"),
        feature_names=splits["X"].columns.tolist(),
        class_names=class_names,
        filled=True,
        rounded=True,
    )

    return results
