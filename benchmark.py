"""
Benchmark de questoes de vestibular usando Groq Cloud.
Compara Llama 3.1 8B vs Llama 3.3 70B (ambos da Meta, na Groq).

Uso:
    export GROQ_API_KEY="sua-chave"
    python benchmark.py dataset.csv

    # Rodar so um modelo:
    python benchmark.py dataset.csv --modelos llama-8b
    python benchmark.py dataset.csv --modelos llama-70b

    # Limitar a 10 questoes (teste rapido):
    python benchmark.py dataset.csv --limite 10

    # Trocar prompt sem perder progresso anterior:
    python benchmark.py dataset.csv --tag prompt_v2

Resume automaticamente de onde parou usando checkpoint.json.
"""

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

# Carrega variaveis do .env (se existir). Variaveis ja setadas no ambiente
# tem prioridade, entao da pra usar tanto .env quanto `export GROQ_API_KEY=...`.
load_dotenv()


# =============================================================================
# CONFIGURACAO
# =============================================================================

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
    "llama-3.1-8b-instant":    1024,
    "llama-3.3-70b-versatile": 1024,
}

SLEEP_BETWEEN_CALLS = 2.1   # free tier = 30 RPM por modelo
MAX_RETRIES = 5

RESULTS_FILE    = "resultados.csv"
CHECKPOINT_FILE = "checkpoint.json"
LOG_FILE        = "benchmark.log"


# =============================================================================
# PROMPT - EDITE AQUI PARA EXPERIMENTAR
# =============================================================================

PROMPT_TEMPLATE = """Voce e um especialista em vestibular brasileiro (ENEM).

QUESTAO {numero} ({campo} - {ano}):
{enunciado}

ALTERNATIVAS:
{alternativas}

Responda APENAS com a letra da alternativa correta (A, B, C, D ou E). Nao explique.

RESPOSTA:"""


# =============================================================================
# UTILITARIOS
# =============================================================================

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
    """Extrai A-E (ou INDETERMINADA) da resposta.

    Suporta varios formatos:
    1. JSON completo com chave 'alternativa_final' (prompt CoT/4+1)
    2. Regex direto pra 'alternativa_final' mesmo em JSON truncado/malformado
    3. 'RESPOSTA: X' (prompt seco)
    4. Ultima letra A-E na string (fallback)
    """
    if not texto:
        return None

    # 1. Tenta JSON completo (formato estruturado)
    try:
        clean = str(texto).strip()
        # Remove code fences ```json ... ```
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```\s*$", "", clean)
        # Pega do primeiro { ate o ultimo }
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                for key in ("alternativa_final", "alternativa", "resposta", "answer"):
                    if key in data:
                        val = str(data[key]).strip().upper()
                        if "INDETERMINADA" in val:
                            return "INDETERMINADA"
                        for c in val:
                            if c in "ABCDE":
                                return c
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 2. Regex direto pra "alternativa_final": "X" (mesmo se JSON malformado/truncado)
    m = re.search(
        r'"\s*alternativa[_\s]?final\s*"\s*:\s*"([A-Ea-e]|INDETERMINADA|indeterminada)',
        str(texto),
    )
    if m:
        val = m.group(1).upper()
        if val == "INDETERMINADA":
            return "INDETERMINADA"
        return val

    txt = str(texto).upper()

    # 3. RESPOSTA: X (prompt seco)
    m = re.search(r"RESPOSTA\s*[:\-]?\s*\(?\s*([A-E])\b", txt)
    if m:
        return m.group(1)

    # 4. INDETERMINADA solta
    if "INDETERMINADA" in txt:
        return "INDETERMINADA"

    # 5. Ultima letra A-E isolada (geralmente a conclusao)
    matches = re.findall(r"\b([A-E])\b", txt)
    if matches:
        return matches[-1]

    # 6. Primeira letra A-E qualquer
    for c in txt:
        if c in "ABCDE":
            return c
    return None


def separar_alternativas(texto):
    """Separa 'a) ... b) ... c) ...' em dict {A: ..., B: ..., ...}.

    Usado quando o prompt tem placeholders separados ({ALTERNATIVA_A}, ...).
    """
    vazio = {"A": "", "B": "", "C": "", "D": "", "E": ""}
    if texto is None or (isinstance(texto, float) and pd.isna(texto)):
        return vazio
    texto = str(texto).strip()
    if not texto:
        return vazio

    # Encontra "a)", "A)", "a.", "A-" etc no comeco de linha
    padrao = r"(?:^|\n)\s*([a-eA-E])\s*[\)\.\-:]\s*"
    matches = list(re.finditer(padrao, texto))
    if not matches:
        return vazio

    resultado = dict(vazio)
    for i, m in enumerate(matches):
        letra = m.group(1).upper()
        inicio = m.end()
        fim = matches[i + 1].start() if i + 1 < len(matches) else len(texto)
        if letra in resultado:
            resultado[letra] = texto[inicio:fim].strip()
    return resultado


def montar_prompt(template, row):
    """Substitui placeholders no template do prompt.

    Suporta dois conjuntos de placeholders:
    - Maiusculos (CoT): {ENUNCIADO_DA_QUESTAO}, {ALTERNATIVA_A..E}
    - Minusculos (seco): {enunciado}, {alternativas}, {numero}, {ano}, {campo}
    """
    enunciado = str(row["enunciado"]).strip()
    alternativas_str = str(row["alternativas"]).strip()
    alts = separar_alternativas(alternativas_str)

    substituicoes = {
        # Formato CoT (placeholders maiusculos)
        "{ENUNCIADO_DA_QUESTAO}": enunciado,
        "{ALTERNATIVA_A}":        alts["A"],
        "{ALTERNATIVA_B}":        alts["B"],
        "{ALTERNATIVA_C}":        alts["C"],
        "{ALTERNATIVA_D}":        alts["D"],
        "{ALTERNATIVA_E}":        alts["E"],
        # Formato seco (placeholders minusculos)
        "{enunciado}":    enunciado,
        "{alternativas}": alternativas_str,
        "{numero}":       str(row["numero_questao"]),
        "{ano}":          str(row["ano"]),
        "{campo}":        str(row["campo"]),
    }

    prompt = template
    for k, v in substituicoes.items():
        prompt = prompt.replace(k, v)
    return prompt


def normalizar_correta(resp):
    """'A)' -> 'A', 'a' -> 'A', etc."""
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


# =============================================================================
# CHAMADA A API
# =============================================================================

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
            espera = 2 ** (tentativa + 2)  # 4, 8, 16, 32, 64
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


# =============================================================================
# MAIN
# =============================================================================

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
    ap.add_argument("--prompt-file", default=None,
                    help="Caminho pra um arquivo de prompt .txt. "
                         "Se nao passar, usa o template embutido (seco).")
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

    # Carrega template do prompt
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            print(f"ERRO: arquivo de prompt nao encontrado: {args.prompt_file}")
            sys.exit(1)
        template = prompt_path.read_text(encoding="utf-8")
        prompt_origem = args.prompt_file
    else:
        template = PROMPT_TEMPLATE
        prompt_origem = "embutido (seco)"

    log(f"=== BENCHMARK iniciado | tag={args.tag} | prompt={prompt_origem} ===")
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
        prompt = montar_prompt(template, row)

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