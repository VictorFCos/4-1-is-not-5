"""
Gera graficos dos resultados do benchmark.

Uso:
    python grafico.py                          # painel completo (default)
    python grafico.py --tipo geral             # so acuracia geral
    python grafico.py --tipo campo             # so por campo
    python grafico.py --tipo ano               # so por ano
    python grafico.py --tipo tokens            # tokens medios por modelo
    python grafico.py --tipo latencia          # distribuicao de latencia
    python grafico.py --tipo custo             # custo total
    python grafico.py --tipo tags              # comparar tags (CoT vs direto)
    python grafico.py --tag direto             # filtrar por tag
    python grafico.py --saida meu.png          # nome customizado
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = "resultados.csv"

PRICES = {
    "llama-3.1-8b-instant":    {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
}

CORES = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def cores_modelos(modelos):
    """Mapeia cada modelo a uma cor consistente."""
    return {m: CORES[i % len(CORES)] for i, m in enumerate(sorted(modelos))}


def custo_linha(row):
    p = PRICES.get(row["modelo_id"])
    if not p:
        return 0.0
    return (row["prompt_tokens"]     * p["input"] +
            row["completion_tokens"] * p["output"]) / 1_000_000


# =============================================================================
# GRAFICOS INDIVIDUAIS
# =============================================================================

def g_geral(ax, df, cmap):
    """Acuracia geral por modelo (barra simples)."""
    acc = df.groupby("modelo_apelido")["acertou"].mean() * 100
    n_q = df.groupby("modelo_apelido")["id"].nunique()

    modelos = acc.index.tolist()
    valores = acc.values
    cores = [cmap[m] for m in modelos]

    bars = ax.bar(modelos, valores, color=cores, edgecolor="white", linewidth=1)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=11, fontweight="bold")

    ax.set_ylabel("Acuracia (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Acuracia geral por modelo", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    rotulos = [f"{m}\n(n={n_q[m]})" for m in modelos]
    ax.set_xticks(range(len(modelos)))
    ax.set_xticklabels(rotulos)


def _grafico_agrupado(ax, df, dim, titulo, cmap):
    """Barras agrupadas por uma dimensao (campo ou ano)."""
    pivot = df.pivot_table(
        index=dim, columns="modelo_apelido",
        values="acertou", aggfunc="mean"
    ) * 100
    pivot = pivot.sort_index()
    n_dim = df.groupby(dim)["id"].nunique()
    acc_geral = df.groupby("modelo_apelido")["acertou"].mean() * 100

    categorias = pivot.index.tolist()
    modelos = pivot.columns.tolist()
    x = np.arange(len(categorias))
    largura = 0.8 / len(modelos)

    for i, modelo in enumerate(modelos):
        offset = (i - (len(modelos) - 1) / 2) * largura
        bars = ax.bar(x + offset, pivot[modelo].values, largura,
                      label=f"{modelo} (geral: {acc_geral[modelo]:.1f}%)",
                      color=cmap[modelo], edgecolor="white", linewidth=0.7)
        ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=8)
        ax.axhline(acc_geral[modelo], color=cmap[modelo],
                   linestyle="--", linewidth=0.7, alpha=0.4)

    rotulos = [f"{c}\n(n={n_dim[c]})" for c in categorias]
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos, fontsize=9)
    ax.set_ylabel("Acuracia (%)")
    ax.set_ylim(0, 110)
    ax.set_title(titulo, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def g_campo(ax, df, cmap):
    _grafico_agrupado(ax, df, "campo", "Acuracia por campo", cmap)


def g_ano(ax, df, cmap):
    if df["ano"].nunique() <= 1:
        ax.text(0.5, 0.5, "(so um ano no dataset)", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray")
        ax.set_title("Acuracia por ano", fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        return
    _grafico_agrupado(ax, df, "ano", "Acuracia por ano", cmap)


def g_tokens(ax, df, cmap):
    """Tokens medios: prompt + completion empilhados por modelo."""
    agg = df.groupby("modelo_apelido").agg(
        prompt=("prompt_tokens", "mean"),
        completion=("completion_tokens", "mean"),
    )
    modelos = agg.index.tolist()
    cores = [cmap[m] for m in modelos]

    ax.bar(modelos, agg["prompt"], color=cores, alpha=0.5, label="prompt (input)")
    ax.bar(modelos, agg["completion"], bottom=agg["prompt"], color=cores,
           label="completion (output)", edgecolor="white", linewidth=1)

    for i, m in enumerate(modelos):
        ax.text(i, agg.loc[m, "prompt"] / 2,
                f"{agg.loc[m, 'prompt']:.0f}",
                ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
        total = agg.loc[m, "prompt"] + agg.loc[m, "completion"]
        ax.text(i, total + total*0.02,
                f"+{agg.loc[m, 'completion']:.1f} comp.",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Tokens medios por chamada")
    ax.set_title("Tokens medios (prompt + completion)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def g_latencia(ax, df, cmap):
    """Boxplot de latencia por modelo."""
    modelos = sorted(df["modelo_apelido"].unique())
    dados = [df[df["modelo_apelido"] == m]["latencia_segundos"].values
             for m in modelos]
    cores = [cmap[m] for m in modelos]

    bp = ax.boxplot(dados, patch_artist=True, widths=0.5)
    ax.set_xticks(range(1, len(modelos) + 1))
    ax.set_xticklabels(modelos)
    for patch, cor in zip(bp["boxes"], cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.7)
    for line in bp["medians"]:
        line.set_color("black")
        line.set_linewidth(1.5)

    for i, m in enumerate(modelos):
        med = np.median(dados[i])
        ax.text(i + 1, med, f"  med={med:.2f}s", va="center",
                fontsize=9, fontweight="bold")

    ax.set_ylabel("Latencia (s)")
    ax.set_title("Distribuicao de latencia por modelo", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def g_custo(ax, df, cmap):
    """Custo total em USD por modelo."""
    df = df.copy()
    df["custo_usd"] = df.apply(custo_linha, axis=1)
    custo = df.groupby("modelo_apelido")["custo_usd"].sum()
    modelos = custo.index.tolist()
    cores = [cmap[m] for m in modelos]

    bars = ax.bar(modelos, custo.values, color=cores,
                  edgecolor="white", linewidth=1)
    ax.bar_label(bars, fmt="$%.4f", padding=3, fontsize=10, fontweight="bold")

    ax.set_ylabel("Custo total (USD)")
    ax.set_title(f"Custo total ({len(df)} chamadas)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def g_tags(ax, df, cmap):
    """Compara acuracia entre tags (ex: prompt CoT vs direto)."""
    if df["tag"].nunique() <= 1:
        tag_unica = df["tag"].iloc[0] if len(df) else "(sem dados)"
        ax.text(0.5, 0.5,
                f"so a tag '{tag_unica}' nos dados\n"
                f"rode com tags diferentes pra comparar",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray")
        ax.set_title("Comparacao entre tags (prompts)", fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        return

    pivot = df.pivot_table(index="tag", columns="modelo_apelido",
                           values="acertou", aggfunc="mean") * 100
    tags = pivot.index.tolist()
    modelos = pivot.columns.tolist()
    x = np.arange(len(tags))
    largura = 0.8 / len(modelos)

    for i, modelo in enumerate(modelos):
        offset = (i - (len(modelos) - 1) / 2) * largura
        bars = ax.bar(x + offset, pivot[modelo].values, largura,
                      label=modelo, color=cmap[modelo],
                      edgecolor="white", linewidth=0.7)
        ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(tags, fontsize=10)
    ax.set_xlabel("Tag (experimento de prompt)")
    ax.set_ylabel("Acuracia (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Acuracia por prompt (tag)", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


TIPOS = {
    "geral":    g_geral,
    "campo":    g_campo,
    "ano":      g_ano,
    "tokens":   g_tokens,
    "latencia": g_latencia,
    "custo":    g_custo,
    "tags":     g_tags,
}


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tipo", default="todos",
                    choices=["todos"] + list(TIPOS.keys()))
    ap.add_argument("--tag", default=None, help="Filtrar por tag")
    ap.add_argument("--saida", default=None, help="Arquivo PNG de saida")
    args = ap.parse_args()

    if not Path(RESULTS_FILE).exists():
        print(f"Arquivo {RESULTS_FILE} nao encontrado. Rode benchmark.py primeiro.")
        return

    df = pd.read_csv(RESULTS_FILE)
    if args.tag:
        df = df[df["tag"] == args.tag]
        if df.empty:
            print(f"Sem dados para tag={args.tag}")
            return

    cmap = cores_modelos(df["modelo_apelido"].unique())

    titulo_tag = args.tag if args.tag else "todas as tags"
    sub = (f"{len(df)} chamadas  |  {df['id'].nunique()} questoes  |  "
           f"tag: {titulo_tag}")

    if args.tipo == "todos":
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        ordem = ["geral", "campo", "ano", "tokens", "latencia", "custo", "tags"]
        for ax, tipo in zip(axes.flat, ordem):
            TIPOS[tipo](ax, df, cmap)
        for ax in axes.flat[len(ordem):]:
            ax.set_visible(False)
        fig.suptitle(f"Benchmark Groq Cloud  -  {sub}",
                     fontsize=14, fontweight="bold", y=0.995)
        saida = args.saida or "graficos_painel.png"
    else:
        fig, ax = plt.subplots(figsize=(11, 6))
        TIPOS[args.tipo](ax, df, cmap)
        fig.suptitle(sub, fontsize=10, color="gray", y=0.99)
        saida = args.saida or f"grafico_{args.tipo}.png"

    plt.tight_layout()
    plt.savefig(saida, dpi=140, bbox_inches="tight")
    print(f"Grafico salvo em: {saida}")
    plt.show()


if __name__ == "__main__":
    main()