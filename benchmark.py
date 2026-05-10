

import os
import sys
import json
import time
import csv
import argparse
import re
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd
    from groq import Groq, APIError, RateLimitError
    from dotenv import load_dotenv
except ImportError:
    print("ERRO: Instale as dependencias: pip install -r requirements.txt")
    sys.exit(1)


load_dotenv()




MODELS = {
    "llama-8b":  "llama-3.1-8b-instant",      # 8B  - producao
    "llama-70b": "llama-3.3-70b-versatile",   # 70B - producao
}

# Precos por 1M tokens (USD) - https://console.groq.com/docs/models
PRICES = {
    "llama-3.1-8b-instant":    {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
}

MAX_TOKENS = {
    "llama-3.1-8b-instant":    64,
    "llama-3.3-70b-versatile": 64,
}

SLEEP_BETWEEN_CALLS = 2.1   # free tier = 30 RPM por modelo
MAX_RETRIES = 5

RESULTS_FILE    = "resultados.csv"
CHECKPOINT_FILE = "checkpoint.json"
LOG_FILE        = "benchmark.log"




PROMPT_TEMPLATE = """Voce e um especialista em vestibular brasileiro (ENEM).

QUESTAO {numero} ({campo} - {ano}):
{enunciado}

ALTERNATIVAS:
{alternativas}

Responda APENAS com a letra da alternativa correta (A, B, C, D ou E). Nao explique.

RESPOSTA:"""



def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def carregar_dataset(caminho):
    """Le csv, tsv ou xlsx automaticamente."""
    p = Path(caminho)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho}")

    suffix = p.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    elif suffix == ".tsv":
        df = pd.read_csv(p, sep="\t")
    else:
        df = pd.read_csv(p)

    obrigatorias = ["id", "campo", "ano", "numero_questao",
                    "enunciado", "alternativas", "resposta_correta"]
    faltantes = [c for c in obrigatorias if c not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas faltantes: {faltantes}\n"
                         f"Encontradas: {list(df.columns)}")
    return df


def extrair_letra(texto):
    
    if not texto:
        return None
    txt = str(texto).upper()

    
    m = re.search(r"RESPOSTA\s*[:\-]?\s*\(?\s*([A-E])\b", txt)
    if m:
        return m.group(1)

  
    matches = re.findall(r"\b([A-E])\b", txt)
    if matches:
        return matches[-1]

    for c in txt:
        if c in "ABCDE":
            return c
    return None


def normalizar_correta(resp):
    
    if pd.isna(resp):
        return None
    s = str(resp).upper().strip()
    for c in s:
        if c in "ABCDE":
            return c
    return None


def carregar_checkpoint():
    if not Path(CHECKPOINT_FILE).exists():
        return set()
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(tuple(k) for k in data.get("done", []))
    except Exception as e:
        log(f"Aviso: checkpoint corrompido ({e}). Comecando do zero.")
        return set()


def salvar_checkpoint(done_set):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"done": [list(t) for t in done_set]}, f,
                  ensure_ascii=False, indent=2)


FIELDNAMES = [
    "timestamp", "tag", "id", "modelo_apelido", "modelo_id",
    "campo", "ano", "numero_questao",
    "resposta_correta", "resposta_modelo", "acertou",
    "prompt_tokens", "completion_tokens", "total_tokens",
    "latencia_segundos", "finish_reason",
    "resposta_bruta", "reasoning",
]


def append_resultado(record):
    file_exists = Path(RESULTS_FILE).exists()
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)



def consultar_modelo(client, model_id, prompt):
    """Faz a chamada e retorna metricas."""
    inicio = time.time()

    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_TOKENS.get(model_id, 1024),
    )
    elapsed = time.time() - inicio

    msg = response.choices[0].message
    content = (msg.content or "").strip()
    reasoning = getattr(msg, "reasoning", None) or ""

    return {
        "resposta_bruta":     content,
        "reasoning":          reasoning,
        "prompt_tokens":      response.usage.prompt_tokens,
        "completion_tokens":  response.usage.completion_tokens,
        "total_tokens":       response.usage.total_tokens,
        "latencia_segundos":  round(elapsed, 3),
        "finish_reason":      response.choices[0].finish_reason,
    }


def consultar_com_retry(client, model_id, prompt):
    for tentativa in range(MAX_RETRIES):
        try:
            return consultar_modelo(client, model_id, prompt)
        except RateLimitError:
            espera = 2 ** (tentativa + 2)  
            log(f"  [rate limit] aguardando {espera}s...")
            time.sleep(espera)
        except APIError as e:
            espera = 2 ** tentativa
            log(f"  [API erro {e}] aguardando {espera}s...")
            time.sleep(espera)
        except Exception as e:
            log(f"  [erro {type(e).__name__}: {e}] tentativa {tentativa+1}/{MAX_RETRIES}")
            time.sleep(2 ** tentativa)
    raise RuntimeError(f"Falhou apos {MAX_RETRIES} tentativas")




def main():
    ap = argparse.ArgumentParser(description="Benchmark Groq Cloud em vestibular")
    ap.add_argument("dataset", help="Caminho do .csv (ou .xlsx) com as questoes")
    ap.add_argument("--modelos", nargs="+",
                    default=list(MODELS.keys()),
                    choices=list(MODELS.keys()),
                    help="Quais modelos rodar")
    ap.add_argument("--limite", type=int, default=None,
                    help="Limita N questoes (para teste rapido)")
    ap.add_argument("--filtrar-campo", default=None,
                    help="Roda so questoes de um campo (ex: Linguagens)")
    ap.add_argument("--tag", default="default",
                    help="Tag pro experimento. Trocar a tag = re-rodar tudo "
                         "(util pra testar prompts diferentes)")
    args = ap.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("ERRO: GROQ_API_KEY nao configurada.")
        print("")
        print("Opcao 1 (recomendada): use o arquivo .env")
        print("  cp .env.example .env")
        print("  # depois edite .env e cole sua chave entre as aspas")
        print("")
        print("Opcao 2: variavel de ambiente direta")
        print("  export GROQ_API_KEY='sua-chave'    # Linux/Mac")
        print("  set GROQ_API_KEY=sua-chave         # Windows cmd")
        print("")
        print("Pegue a chave gratuita em: https://console.groq.com/keys")
        sys.exit(1)

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    log(f"=== BENCHMARK iniciado | tag={args.tag} ===")
    log(f"Carregando: {args.dataset}")
    df = carregar_dataset(args.dataset)
    log(f"Total no dataset: {len(df)} linhas")

    if args.filtrar_campo:
        df = df[df["campo"].astype(str)
                .str.contains(args.filtrar_campo, case=False, na=False)]
        log(f"Apos filtro '{args.filtrar_campo}': {len(df)}")

    if args.limite:
        df = df.head(args.limite)
        log(f"Limitado a: {len(df)}")

    done = carregar_checkpoint()
    log(f"Checkpoint: {len(done)} pares (id, modelo, tag) ja feitos")

    modelos = {k: MODELS[k] for k in args.modelos}
    log(f"Modelos: {modelos}")

    total = len(df) * len(modelos)
    contador = 0
    stats = {m: [0, 0] for m in modelos}  # [acertos, total]

    for _, row in df.iterrows():
        qid = str(row["id"])
        correta = normalizar_correta(row["resposta_correta"])
        prompt = PROMPT_TEMPLATE.format(
            numero=row["numero_questao"],
            campo=row["campo"],
            ano=row["ano"],
            enunciado=str(row["enunciado"]).strip(),
            alternativas=str(row["alternativas"]).strip(),
        )

        for apelido, model_id in modelos.items():
            chave = (qid, apelido, args.tag)
            contador += 1

            if chave in done:
                continue

            log(f"[{contador}/{total}] {qid} | {apelido}")

            try:
                res = consultar_com_retry(client, model_id, prompt)
            except Exception as e:
                log(f"  ERRO FATAL em {qid}/{apelido}: {e}")
                continue  # nao marca done -> tenta de novo na proxima execucao

            predicted = extrair_letra(res["resposta_bruta"])
            acertou = bool(predicted and correta and predicted == correta)

            stats[apelido][1] += 1
            if acertou:
                stats[apelido][0] += 1

            record = {
                "timestamp":         datetime.now().isoformat(timespec="seconds"),
                "tag":               args.tag,
                "id":                qid,
                "modelo_apelido":    apelido,
                "modelo_id":         model_id,
                "campo":             row["campo"],
                "ano":               row["ano"],
                "numero_questao":    row["numero_questao"],
                "resposta_correta":  correta,
                "resposta_modelo":   predicted,
                "acertou":           acertou,
                "prompt_tokens":     res["prompt_tokens"],
                "completion_tokens": res["completion_tokens"],
                "total_tokens":      res["total_tokens"],
                "latencia_segundos": res["latencia_segundos"],
                "finish_reason":     res["finish_reason"],
                "resposta_bruta":    res["resposta_bruta"],
                "reasoning":         (res["reasoning"] or "")[:2000],
            }
            append_resultado(record)
            done.add(chave)
            salvar_checkpoint(done)

            ac, tot = stats[apelido]
            taxa = (ac / tot * 100) if tot else 0
            simb = "OK" if acertou else "X "
            log(f"  [{simb}] resp={predicted} (correta={correta})  "
                f"tokens={res['total_tokens']}  "
                f"lat={res['latencia_segundos']}s  "
                f"acc parcial={ac}/{tot} ({taxa:.1f}%)")

            time.sleep(SLEEP_BETWEEN_CALLS)

    log("\n" + "=" * 60)
    log("BENCHMARK CONCLUIDO")
    log("=" * 60)
    for apelido, (ac, tot) in stats.items():
        if tot:
            log(f"  {apelido}: {ac}/{tot} ({ac/tot*100:.1f}%)")
        else:
            log(f"  {apelido}: nenhuma chamada nova (tudo em checkpoint)")
    log(f"\nResultados em: {RESULTS_FILE}")
    log(f"Para analise:  python analise.py")


if __name__ == "__main__":
    main()