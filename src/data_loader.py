from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Carrega dataset sem cabeçalho e separado por ponto e vírgula.
    A última coluna é considerada a coluna alvo (target) e as demais são consideradas features.
    As colunas são nomeadas como feature_01, feature_02, ..., feature_N, target.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")
    df = pd.read_csv(path, sep=";", header=None)
    if df.shape[1] < 2:
        raise ValueError("O dataset deve possuir pelo menos uma feature e uma coluna alvo.")
    df.columns = [f"feature_{i + 1:02d}" for i in range(df.shape[1] - 1)] + ["target"]
    return df


def infer_dataset_profile(df: pd.DataFrame) -> dict:
    """
    Analisa o dataset e retorna um dicionário com informações sobre a estrutura, tipos de atributos, distribuição das classes, presença de valores nulos, duplicados, variância dos atributos, escala dos dados e recomendação de normalização. O perfil inclui:
    - Número de linhas e colunas    
    - Número de atributos (excluindo a coluna alvo)
    - Nome da coluna alvo    
    - Tipos de atributos (numéricos e não numéricos)
    - Classes presentes na coluna alvo e suas contagens    
    - Tipo de problema (binário ou multiclasse)
    - Presença de duplicados    
    - Presença de valores nulos    
    - Atributos com variância zero    
    - Razão de escala entre os atributos numéricos    
    - Recomendação de normalização (se a razão de escala for maior que 10)
    """
    feature_df = df.drop(columns=["target"])
    numeric_features = feature_df.apply(pd.to_numeric, errors="coerce")
    non_numeric_columns = numeric_features.columns[numeric_features.isna().any() & feature_df.notna().any()].tolist()
    target_counts = df["target"].astype(str).value_counts()
    imbalance_ratio = float(target_counts.max() / target_counts.min()) if len(target_counts) > 1 else np.inf
    numeric_ranges = (numeric_features.max() - numeric_features.min()).replace([np.inf, -np.inf], np.nan)
    scale_ratio = float(numeric_ranges.max() / numeric_ranges[numeric_ranges > 0].min())
    zero_variance = numeric_features.columns[numeric_features.nunique(dropna=True) <= 1].tolist()
    outlier_counts = {}
    for col in numeric_features.columns:
        q1 = numeric_features[col].quantile(0.25)
        q3 = numeric_features[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            outlier_counts[col] = 0
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_counts[col] = int(((numeric_features[col] < lower) | (numeric_features[col] > upper)).sum())
    return {
        "n_linhas": int(df.shape[0]),
        "n_colunas_total": int(df.shape[1]),
        "n_atributos": int(feature_df.shape[1]),
        "coluna_alvo": "target",
        "tipos_atributos": {
            "numericos": int(len(feature_df.columns) - len(non_numeric_columns)),
            "nao_numericos": int(len(non_numeric_columns)),
            "colunas_nao_numericas": non_numeric_columns,
        },
        "classes": target_counts.index.tolist(),
        "distribuicao_classes": target_counts.to_dict(),
        "problema": "binario" if len(target_counts) == 2 else "multiclasse",
        "razao_desbalanceamento": imbalance_ratio,
        "ha_desbalanceamento": bool(imbalance_ratio >= 1.5),
        "valores_nulos_total": int(df.isna().sum().sum()),
        "duplicados": int(df.duplicated().sum()),
        "atributos_variancia_zero": zero_variance,
        "razao_escala_atributos": scale_ratio,
        "normalizacao_recomendada": bool(scale_ratio > 10),
        "outliers_por_atributo": outlier_counts,
        "total_outliers_iqr": int(sum(outlier_counts.values())),
        "padrao_geral": (
            "Dataset tabular sem cabeçalho, composto predominantemente por atributos numéricos "
            "e rótulo categórico na última coluna."
        ),
    }
