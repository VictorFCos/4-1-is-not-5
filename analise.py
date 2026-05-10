import argparse
from pathlib import Path

import pandas as pd

RESULTS_FILE = "resultados.csv"

PRICES = {
    "llama-3.1-8b-instant":    {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
}


def custo_linha(row):
    p = PRICES.get(row["modelo_id"])
    if not p:
        return 0.0
    return (row["prompt_tokens"]     * p["input"] +
            row["completion_tokens"] * p["output"]) / 1_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="Filtrar por tag de experimento")
    ap.add_argument("--export", default=None, help="Salvar resumo em CSV")
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

    df["custo_usd"] = df.apply(custo_linha, axis=1)

    print("=" * 78)
    print(f"RESUMO GERAL  ({len(df)} chamadas | "
          f"{df['id'].nunique()} questoes unicas | "
          f"tags={list(df['tag'].unique())})")
    print("=" * 78)

    resumo = df.groupby("modelo_apelido").agg(
        questoes               = ("id",                "nunique"),
        chamadas               = ("id",                "count"),
        acertos                = ("acertou",           "sum"),
        acuracia_pct           = ("acertou",           lambda s: round(s.mean()*100, 2)),
        prompt_tokens_medio    = ("prompt_tokens",     "mean"),
        completion_tokens_med  = ("completion_tokens", "mean"),
        total_tokens_total     = ("total_tokens",      "sum"),
        latencia_media_s       = ("latencia_segundos", "mean"),
        custo_total_usd        = ("custo_usd",         "sum"),
    ).round(4)
    print(resumo.to_string())


    print("\n" + "=" * 78)
    print("ACURACIA (%) POR CAMPO E MODELO")
    print("=" * 78)
    pivot = df.pivot_table(
        index="campo", columns="modelo_apelido",
        values="acertou", aggfunc="mean"
    ) * 100
    cont = df.groupby("campo")["id"].nunique().rename("n_questoes")
    pivot = pivot.round(2).join(cont)
    print(pivot.fillna("-").to_string())


    if df["ano"].nunique() > 1:
        print("\n" + "=" * 78)
        print("ACURACIA (%) POR ANO E MODELO")
        print("=" * 78)
        pivot_ano = df.pivot_table(
            index="ano", columns="modelo_apelido",
            values="acertou", aggfunc="mean"
        ) * 100
        print(pivot_ano.round(2).to_string())

    print("\n" + "=" * 78)
    print("DISCORDANCIAS  (modelos deram respostas diferentes)")
    print("=" * 78)
    pivot_resp = df.pivot_table(
        index="id", columns="modelo_apelido",
        values="resposta_modelo", aggfunc="first"
    )
    if pivot_resp.shape[1] >= 2:
        col1, col2 = pivot_resp.columns[:2]
        diff = pivot_resp[pivot_resp[col1] != pivot_resp[col2]].copy()
        print(f"Total de discordancias: {len(diff)} de {len(pivot_resp)}")
        if len(diff):
            corretas = df.drop_duplicates("id").set_index("id")["resposta_correta"]
            diff = diff.join(corretas, how="left")
            print("\nPrimeiras 20:")
            print(diff.head(20).to_string())
    else:
        print("(precisa de pelo menos 2 modelos)")


    print("\n" + "=" * 78)
    print("FINISH REASONS  (atencao a 'length' = resposta cortada)")
    print("=" * 78)
    fr = df.groupby(["modelo_apelido", "finish_reason"]).size().unstack(fill_value=0)
    print(fr.to_string())


    print("\n" + "=" * 78)
    print("ERROS DE PARSING  (modelo respondeu mas nao extraimos a letra)")
    print("=" * 78)
    nao_parsed = df[df["resposta_modelo"].isna()]
    print(f"Total: {len(nao_parsed)}")
    if len(nao_parsed):
        print(nao_parsed[["id", "modelo_apelido", "resposta_bruta"]]
              .head(10).to_string())


    print("\n" + "=" * 78)
    print("QUESTOES QUE TODOS OS MODELOS ERRARAM")
    print("=" * 78)
    acertos_por_q = df.groupby("id")["acertou"].sum()
    n_modelos = df["modelo_apelido"].nunique()
    todos_erraram = acertos_por_q[acertos_por_q == 0].index.tolist()
    print(f"Total: {len(todos_erraram)} questoes "
          f"(de {df['id'].nunique()})")
    if todos_erraram:
        amostras = df[df["id"].isin(todos_erraram)].drop_duplicates("id")[
            ["id", "campo", "numero_questao", "resposta_correta"]
        ].head(15)
        print(amostras.to_string(index=False))

    if args.export:
        resumo.to_csv(args.export)
        print(f"\nResumo exportado para: {args.export}")


if __name__ == "__main__":
    main()