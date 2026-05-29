from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def _supports_color() -> bool:
    if not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
                return True
        except Exception:
            return False
    return True


class _C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"


def _c(text: str, *styles: str) -> str:
    if not _supports_color():
        return text
    return "".join(styles) + text + _C.RESET


def print_banner(title: str, subtitle: str = "") -> None:
    width = 72
    line = "=" * width
    print()
    print(_c(line, _C.CYAN, _C.BOLD))
    print(_c(f"  {title}", _C.CYAN, _C.BOLD))
    if subtitle:
        print(_c(f"  {subtitle}", _C.DIM))
    print(_c(line, _C.CYAN, _C.BOLD))
    print()


def print_step(current: int, total: int, message: str, status: str = "ok") -> None:
    label = f"[{current}/{total}]"
    dots = "." * max(1, 40 - len(message))
    if status == "ok":
        mark = _c("OK", _C.GREEN, _C.BOLD)
    elif status == "run":
        mark = _c("...", _C.YELLOW)
    else:
        mark = _c(status, _C.YELLOW)
    print(f"  {_c(label, _C.DIM)} {message} {dots} {mark}")


def print_section(title: str) -> None:
    print()
    print(_c(f"  {title}", _C.BOLD, _C.BLUE))
    print(_c("  " + "-" * 68, _C.DIM))


def print_supervised_table(comparison: pd.DataFrame) -> None:
    print_section("RESULTADOS SUPERVISIONADOS (conjunto de teste)")
    best_name = comparison.iloc[0]["modelo"]
    header = f"  {'Modelo':<24} {'F1-val':>8} {'F1-test':>8} {'Acurácia':>9} {'ROC-AUC':>8}"
    print(_c(header, _C.DIM))
    print(_c("  " + "-" * 66, _C.DIM))
    for _, row in comparison.iterrows():
        name = str(row["modelo"])
        f1_val = row.get("f1_validacao", float("nan"))
        roc = row.get("roc_auc", None)
        roc_str = f"{roc:.3f}" if roc is not None and roc == roc else "  —   "
        line = f"  {name:<24} {f1_val:>8.3f} {row['f1']:>8.3f} {row['accuracy']:>9.3f} {roc_str:>8}"
        if name == best_name:
            print(_c(line + "  <- melhor F1 (teste)", _C.GREEN, _C.BOLD))
        else:
            print(line)


def print_kmeans_summary(unsupervised: dict) -> None:
    print_section("K-MEANS (treino, rótulos ignorados no ajuste)")
    print(f"  K testados .............. {unsupervised['tested_k']}")
    print(
        f"  Melhor K (silhouette) ... {unsupervised['best_k_silhouette']} "
        f"(silhouette={unsupervised['best_silhouette']:.3f}, "
        f"ARI={unsupervised['best_k_ari']:.3f})"
    )
    print(
        f"  K = nº de classes ....... {unsupervised['k_equals_n_classes']} "
        f"(ARI={unsupervised['k_n_classes_ari']:.3f}, "
        f"NMI={unsupervised['k_n_classes_nmi']:.3f})"
    )


def print_artifacts() -> None:
    print_section("ARTEFATOS GERADOS")
    paths = [
        ("Figuras", "results/figures/"),
        ("Métricas", "results/metrics/"),
        ("Modelos", "results/models/"),
        ("Relatório", "report/relatorio.md"),
    ]
    for label, path in paths:
        print(f"  {label:<12} -> {path}")


def print_footer(elapsed_s: float) -> None:
    width = 72
    print()
    print(_c("=" * width, _C.GREEN, _C.BOLD))
    print(_c("  Pipeline concluído com sucesso.", _C.GREEN, _C.BOLD))
    print(_c(f"  Tempo total: {elapsed_s:.1f}s", _C.DIM))
    print(_c("=" * width, _C.GREEN, _C.BOLD))
    print()
