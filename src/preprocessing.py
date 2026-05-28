from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils import RANDOM_STATE


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza a limpeza do dataset, convertendo as colunas de features para numéricas (substituindo valores não numéricos por NaN), padronizando os rótulos da coluna alvo, preenchendo valores ausentes com a mediana das colunas numéricas e removendo duplicatas. O resultado é um DataFrame limpo e pronto para análise e modelagem.
    Args:
        df (pd.DataFrame): O DataFrame original a ser limpo.
    Returns:
        pd.DataFrame: O DataFrame limpo e pronto para análise e modelagem.
    """
    cleaned = df.copy()
    for col in cleaned.columns[:-1]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned["target"] = cleaned["target"].astype(str).str.strip()
    if cleaned.iloc[:, :-1].isna().any().any():
        cleaned.iloc[:, :-1] = cleaned.iloc[:, :-1].fillna(cleaned.iloc[:, :-1].median())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def prepare_splits(df: pd.DataFrame, test_size: float = 0.20, validation_size: float = 0.20) -> dict:
    """
    Prepara os conjuntos de treino, validação e teste a partir do DataFrame limpo, realizando a codificação dos rótulos da coluna alvo, a padronização das features numéricas e a divisão estratificada dos dados. O resultado é um dicionário contendo os conjuntos de dados prontos para modelagem, incluindo as versões escaladas para KNN e MLP.

    Args:
        df (pd.DataFrame): O DataFrame limpo contendo os dados para modelagem.
        test_size (float, optional): O tamanho da parte de teste. Defaults to 0.20.
        validation_size (float, optional): O tamanho da parte de validação. Defaults to 0.20.

    Returns:
        dict: Um dicionário contendo os conjuntos de dados prontos para modelagem, incluindo as versões escaladas para KNN e MLP.
    """
    X = df.drop(columns=["target"])
    y_raw = df["target"]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    relative_validation = validation_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_validation,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    # A padronização é indispensável para KNN e RNA/MLP porque ambos dependem de magnitudes:
    # o KNN calcula distâncias no espaço de atributos, e a MLP otimiza pesos por gradiente.
    # Sem escala comum, atributos com valores maiores dominam distâncias e atualizações.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "split_info": {
            "treino": round(len(X_train) / len(X), 3),
            "validacao": round(len(X_val) / len(X), 3),
            "teste": round(len(X_test) / len(X), 3),
            "stratify": True,
            "random_state": RANDOM_STATE,
        },
    }
