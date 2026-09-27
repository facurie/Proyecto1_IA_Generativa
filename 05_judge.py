# %% [markdown]
r"""
# Etapa 5 · Medir la emergencia — un juez local

Un número de perplejidad no dice *qué* mejoró. Acá un modelo local y gratuito (Ollama + `qwen3:4b`, sin
modo de razonamiento) le pone nota, como un docente corrigiendo a un alumno, a generaciones de:

- **base (ancho)** y **profundo**: las dos variantes de la ablación de la Etapa 2;
- **SFT completo** y **SFT LoRA**: los dos modelos de la Etapa 4 (LoRA, si se corrió).

Criterios, de 1 a 10: **gramática**, **creatividad**, **consistencia** y, para los prompts con instrucción,
**obediencia**. La obediencia también se le pide al base sobre los mismos prompts: es el control que dice
cuánto de "seguir la instrucción" ya estaba antes del SFT.

**Anclas para calibrar al juez** (nadie las pidió, pero sin ellas no se sabe si el juez discrimina):
cuentos **reales** de TinyStories (el techo esperable), los mismos cuentos con las **palabras mezcladas**
(misma temática y vocabulario, gramática destruida) y cuentos reales evaluados contra una **instrucción
ajena** (obediencia debería dar baja). Si el juez no separa esas tres cosas, sus notas sobre nuestros
modelos no valen mucho.

Después se cruza la lectura del juez con la **perplejidad** y con los prompts de prueba de la Etapa 2.

**Lee:** `checkpoints/tokenizer.json`, `pretrain_wide.pt`, `pretrain_deep.pt` (Etapa 2), `sft_final.pt`,
`sft_lora.pt` (Etapa 4; el de LoRA y el profundo son opcionales), y si existen `stage2_ablation.csv` y
`stage4/comparacion_lora.csv`. **Produce:** `checkpoints/stage5/` (notas, tablas, gráficos). Las respuestas
del juez se guardan a medida que llegan (`juez_respuestas.jsonl`): si Colab se corta, volver a correr
retoma sin repetir llamadas.

**Cómo correrlo.** En Colab con GPU: la celda del juez instala Ollama en la VM y baja `qwen3:4b` (~2,5 GB).
En CPU, `SMOKE_TEST` se activa solo y, si no hay Ollama, usa un **juez falso** que a propósito devuelve
respuestas mal formadas (dos objetos, texto alrededor, claves faltantes) para probar el parser. Sus notas
no significan nada.
"""

# %%
import environment  # noqa: F401  (primero siempre: caché de HF, stdout UTF-8, checkpoints/)

environment.summary()

import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.nn import functional as F

from data import load_tinystories

DEVICE = environment.device()
SMOKE_TEST = environment.smoke_test(DEVICE == "cpu")  # CPU: validar el pipeline. GPU: la corrida de verdad. LAB_SMOKE_TEST fuerza.
SEED = 1337

CKPT_DIR = environment.CHECKPOINTS / "smoke" if SMOKE_TEST else environment.CHECKPOINTS
TOKENIZER_PATH = environment.CHECKPOINTS / "tokenizer.json"
OUTPUT_DIR = CKPT_DIR / "stage5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Nombre legible -> checkpoint. Los obligatorios son los que pide la consigna como mínimo (Etapas 2 y 4).
MODEL_PATHS = {
    "base (ancho)": CKPT_DIR / "pretrain_wide.pt",
    "profundo": CKPT_DIR / "pretrain_deep.pt",
    "SFT completo": CKPT_DIR / "sft_final.pt",
    "SFT LoRA": CKPT_DIR / "sft_lora.pt",
}
REQUIRED = {"base (ancho)", "SFT completo"}

JUDGE_MODEL = "qwen3:4b"
USE_AMP = DEVICE == "cuda"

torch.manual_seed(SEED)
print(f"dispositivo: {DEVICE} · SMOKE_TEST: {SMOKE_TEST} · checkpoints en: {CKPT_DIR}")

# %% [markdown]
r"""
## Los modelos

Mismas clases que en las Etapas 2 y 4 (copia textual: importar esos archivos los ejecutaría enteros). El de
LoRA se guardó ya fusionado en la Etapa 4, así que se carga igual que los demás.
"""


# %%
@dataclass
class GPTConfig:
    """Todo lo que define la FORMA del modelo. Va en cada checkpoint para reconstruirlo sin adivinar."""

    vocab_size: int
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float = 0.1


class Head(nn.Module):
    """Una cabeza de self-attention causal."""

    def __init__(self, cfg: GPTConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)), persistent=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)  # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ self.value(x)  # (B, T, hs)


class MultiHeadAttention(nn.Module):
    """Varias cabezas en paralelo; sus salidas se concatenan y se proyectan de vuelta a n_embd."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList(Head(cfg, head_size) for _ in range(cfg.n_head))
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """MLP por posición: expande a 4·n_embd, no linealidad, vuelve a n_embd."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Comunicación (atención) y después cómputo (MLP), con pre-LayerNorm y conexiones residuales."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.sa = MultiHeadAttention(cfg)
        self.ffwd = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class GPTLanguageModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd tiene que ser múltiplo de n_head"
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding_table(idx) + self.position_embedding_table(pos)  # (B, T, n_embd)
        logits = self.lm_head(self.ln_f(self.blocks(x)))  # (B, T, V)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1).float(), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, *, temperature=1.0, greedy=False, generator=None, stop_token=None):
        """Extiende `idx` (B, T) token a token. Con `greedy`, siempre el más probable; si no, muestrea."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size :])
            logits = logits[:, -1, :].float()
            if greedy:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_token is not None and (next_id == stop_token).all():
                break
        return idx


def load_model(path, device=DEVICE):
    """Reconstruye un modelo a partir de un checkpoint. Devuelve (modelo en modo eval, checkpoint crudo)."""
    checkpoint = torch.load(path, map_location=device)
    model = GPTLanguageModel(GPTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


# %%
if not TOKENIZER_PATH.is_file():
    raise FileNotFoundError(f"No está {TOKENIZER_PATH}. Corré primero la Etapa 1, o traé el archivo de Drive.")
missing = [f"{name}: {path}" for name, path in MODEL_PATHS.items() if name in REQUIRED and not path.is_file()]
if missing:
    raise FileNotFoundError("Faltan checkpoints obligatorios (Etapas 2 y 4):\n  " + "\n  ".join(missing))

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
VOCAB_SIZE = tokenizer.get_vocab_size()
EOT_ID = tokenizer.token_to_id("<|endoftext|>")
TOKENIZER_SHA1 = hashlib.sha1(TOKENIZER_PATH.read_bytes()).hexdigest()

models, checkpoints = {}, {}
for name, path in MODEL_PATHS.items():
    if not path.is_file():
        print(f"aviso: no está {path.name}; '{name}' queda afuera de la comparación.")
        continue
    models[name], checkpoints[name] = load_model(path)
    assert models[name].cfg.vocab_size == VOCAB_SIZE, f"{path.name} no es del vocabulario de {TOKENIZER_PATH.name}"
    # Checkpoints viejos sin huella se aceptan; uno con huella distinta es de otro tokenizer.
    if checkpoints[name].get("tokenizer_sha1", TOKENIZER_SHA1) != TOKENIZER_SHA1:
        raise ValueError(f"{path.name} se entrenó con otro tokenizer.json (¿se volvió a correr la Etapa 1?).")
    n_params = sum(p.numel() for p in models[name].parameters())
    print(f"{name:<14} {path.name:<18} paso {checkpoints[name]['step']:>6,} · {n_params / 1e6:.2f}M parámetros")

# %% [markdown]
r"""
## Perplejidad de cada modelo, con la misma vara

La pérdida sobre el split `validation` de TinyStories liso, **igual para todos**: los mismos batches (misma
semilla), ventanas al azar como en la Etapa 2. Así la pérdida del SFT (entrenado en otro formato) y la de
los preentrenados se leen en la misma escala. Ojo: para los de SFT es texto *fuera* de su formato de
entrenamiento, así que una suba acá mide olvido, no calidad.
"""

# %%
N_VAL_STORIES = 300 if SMOKE_TEST else 5_000
EVAL_BATCHES, EVAL_BATCH_SIZE = (5, 8) if SMOKE_TEST else (40, 64)

val_texts = list(load_tinystories("validation", limit=N_VAL_STORIES)["text"])
val_ids = np.concatenate([np.array(e.ids + [EOT_ID], dtype=np.int64) for e in tokenizer.encode_batch(val_texts)])


def autocast():
    return torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP)


@torch.no_grad()
def val_loss(model) -> float:
    generator = torch.Generator().manual_seed(SEED)
    T = model.cfg.block_size
    losses = []
    for _ in range(EVAL_BATCHES):
        starts = torch.randint(len(val_ids) - T - 1, (EVAL_BATCH_SIZE,), generator=generator).tolist()
        windows = torch.from_numpy(np.stack([val_ids[i : i + T + 1] for i in starts])).to(DEVICE)
        with autocast():
            _, loss = model(windows[:, :-1].contiguous(), windows[:, 1:].contiguous())
        losses.append(loss.item())
    return float(np.mean(losses))


perplexity = pd.DataFrame(
    {name: {"val loss (TinyStories liso)": (loss := val_loss(m)), "val ppl": math.exp(loss)} for name, m in models.items()}
).T
print(perplexity.to_string(float_format="{:.3f}".format))

# %% [markdown]
r"""
## Las generaciones a evaluar

Dos familias de prompts, fijadas antes de ver notas:

- **cuento**: comienzos de cuento sueltos (el formato de la Etapa 2). Criterios: gramática, creatividad,
  consistencia.
- **instrucción**: `Words: a, b, c` + `Story:` (el formato de la Etapa 4, cortado en `Story:` por la trampa
  del borde de tokenización que se midió ahí). Se suma obediencia.

Varias muestras por prompt (la consigna: con una sola no se separan modelos a esta escala), con semilla
`SEED + 100·prompt + muestra`, idéntica entre modelos. Temperatura 1, como en las Etapas 2 y 4.
"""

# %%
STORY_PROMPTS = [
    "Once upon a time",
    "One day, a little girl named Lily",
    "Tom and his dog went to the park.",
    "There was a big red ball in the garden.",
    "Sam was sad because",
    "The little bird wanted to fly high.",
]
WORD_PROMPTS = [
    ("dragon", "happy", "forest"),
    ("ball", "sad", "garden"),
    ("cat", "brave", "river"),
    ("cake", "angry", "school"),
    ("bird", "scared", "tree"),
    ("boat", "kind", "sea"),
]
if SMOKE_TEST:
    STORY_PROMPTS, WORD_PROMPTS = STORY_PROMPTS[:2], WORD_PROMPTS[:2]
N_SAMPLES = 1 if SMOKE_TEST else 4
MAX_NEW_TOKENS = 200


def instruct_prompt(words) -> str:
    return f"Words: {', '.join(words)}\nStory:"


def generate(model, prompt: str, seed: int) -> str:
    """Continuación de `[EOT] + prompt`, hasta `<|endoftext|>` o MAX_NEW_TOKENS."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    idx = torch.tensor([[EOT_ID] + tokenizer.encode(prompt).ids], device=DEVICE)
    with autocast():
        out = model.generate(idx, MAX_NEW_TOKENS, generator=generator, stop_token=EOT_ID)
    new = out[0, idx.shape[1] :].tolist()
    if EOT_ID in new:
        new = new[: new.index(EOT_ID)]
    return tokenizer.decode(new)


def word_used(word: str, text: str) -> bool:
    """Misma regla que la Etapa 4: la palabra o una flexión con sufijo explícito (`cat` no cuenta `catch`)."""
    forms = rf"{re.escape(word)}(s|es|d|ed|ing|er|est|ly|ness)?"
    if word.endswith("y"):
        forms += rf"|{re.escape(word[:-1])}i(es|ed|er|est|ly|ness)"
    return re.search(rf"\b(?:{forms})\b", text, flags=re.IGNORECASE) is not None


items = []  # una fila por texto a evaluar
for name, model in models.items():
    for i, prompt in enumerate(STORY_PROMPTS):
        for j in range(N_SAMPLES):
            story = prompt + generate(model, prompt, SEED + 100 * i + j)
            items.append({"fuente": name, "tipo": "cuento", "prompt": i, "muestra": j, "texto": story, "palabras": None})
    for i, words in enumerate(WORD_PROMPTS):
        for j in range(N_SAMPLES):
            story = generate(model, instruct_prompt(words), SEED + 100 * i + j).strip()
            items.append({"fuente": name, "tipo": "instrucción", "prompt": i, "muestra": j, "texto": story, "palabras": list(words)})

# Anclas: cuentos reales (del final del split validation, para no pisar los de la perplejidad), mezclados
# palabra por palabra, y reales contra una instrucción ajena.
N_ANCHORS = len(STORY_PROMPTS) * N_SAMPLES
anchor_texts = [t.strip() for t in load_tinystories("validation", limit=N_VAL_STORIES + N_ANCHORS)["text"][N_VAL_STORIES:]]
rng = random.Random(SEED)
for k, text in enumerate(anchor_texts):
    shuffled = text.split()
    rng.shuffle(shuffled)
    words = WORD_PROMPTS[k % len(WORD_PROMPTS)]
    items.append({"fuente": "ancla: cuento real", "tipo": "cuento", "prompt": k, "muestra": 0, "texto": text, "palabras": None})
    items.append({"fuente": "ancla: palabras mezcladas", "tipo": "cuento", "prompt": k, "muestra": 0, "texto": " ".join(shuffled), "palabras": None})
    items.append({"fuente": "ancla: instrucción ajena", "tipo": "instrucción", "prompt": k, "muestra": 0, "texto": text, "palabras": list(words)})

items = pd.DataFrame(items)
items["uso de palabras"] = [
    np.mean([word_used(w, t) for w in ws]) if ws else np.nan for t, ws in zip(items["texto"], items["palabras"])
]
print(items.groupby(["fuente", "tipo"], sort=False).size().unstack().to_string())
print("\nejemplo:", items.iloc[0]["texto"][:300])

# %% [markdown]
r"""
### Perplejidad por texto

Para cruzar juez y perplejidad **texto por texto**, cada texto se puntúa con un mismo modelo de referencia
(el preentrenado de menor pérdida de validación): NLL media por token del texto. Es una vara fija, pero no
neutral: favorece lo que se parece a sus propias muestras. Se informa igual, con esa salvedad.
"""


# %%
@torch.no_grad()
def text_nll(model, text: str) -> float:
    ids = [EOT_ID] + tokenizer.encode(text).ids
    ids = ids[: model.cfg.block_size + 1]
    x = torch.tensor([ids], device=DEVICE)
    with autocast():
        _, loss = model(x[:, :-1].contiguous(), x[:, 1:].contiguous())
    return loss.item()


pretrained = [n for n in ("base (ancho)", "profundo") if n in models]
REFERENCE = min(pretrained, key=lambda n: perplexity.loc[n, "val loss (TinyStories liso)"])
items["NLL (ref)"] = [text_nll(models[REFERENCE], t) if t.strip() else np.nan for t in items["texto"]]
print(f"modelo de referencia para la NLL por texto: {REFERENCE}")
print(items.groupby("fuente", sort=False)["NLL (ref)"].mean().to_string(float_format="{:.3f}".format))

# Liberar la GPU para el juez: a partir de acá no se genera más.
del models
if DEVICE == "cuda":
    torch.cuda.empty_cache()

# %% [markdown]
r"""
## El juez

Ollama se instala en la VM de Colab, se levanta el servidor, se espera a que responda y se baja
`qwen3:4b`. Llamadas con `think=False` (que puntúe directo), `format="json"` (el decoder queda obligado a
devolver JSON válido) y temperatura 0 con semilla fija (la misma entrada da la misma nota).

**Trampa de la consigna: el juez a veces contesta dos veces**, o envuelve el JSON en una oración. El parser
no usa una regex glotona: recorre las llaves con `json.JSONDecoder().raw_decode`, que corta al final del
primer valor, y se queda con el primer objeto que tenga todas las claves con enteros entre 1 y 10. Cada
llamada va envuelta: una respuesta mala se registra como falla (con el texto crudo) y la corrida sigue.
Cuántas veces pasa eso es, en sí, un resultado.
"""


# %%
def ollama_ready() -> bool:
    try:
        import requests

        requests.get("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def start_ollama() -> bool:
    """Instala (solo en Colab), levanta el servidor y baja el modelo. Devuelve False si no hay Ollama."""
    if shutil.which("ollama") is None:
        if "google.colab" not in sys.modules:
            return False
        # El instalador actual descomprime con zstd, que la imagen de Colab no siempre trae.
        subprocess.run("apt-get -qq install -y zstd > /dev/null", shell=True, check=False)
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    if not ollama_ready():
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(30):
            if ollama_ready():
                break
            time.sleep(1)
        else:
            raise RuntimeError("`ollama serve` no respondió en 30 s")
    subprocess.run(["ollama", "pull", JUDGE_MODEL], check=True)
    return True


JUDGE_IS_REAL = start_ollama()
if not JUDGE_IS_REAL:
    if not SMOKE_TEST:
        raise RuntimeError("No hay Ollama. En Colab esta celda lo instala; fuera de Colab, instalalo a mano.")
    print("SMOKE_TEST sin Ollama: se usa un juez FALSO que solo sirve para probar el parser.")
print(f"juez: {JUDGE_MODEL if JUDGE_IS_REAL else 'falso'}")

# %%
CRITERIA = {
    "gramática": "grammar",
    "creatividad": "creativity",
    "consistencia": "consistency",
    "obediencia": "obedience",
}
RUBRIC = {
    "grammar": "Are the sentences grammatical and well formed English?",
    "creativity": "Is the story original and interesting, with some imagination, rather than generic or repetitive?",
    "consistency": "Does the story make sense as a whole: characters, objects and events stay coherent from start to end?",
    "obedience": "Does the story follow the instruction, using all the required words naturally in the story?",
}


def grading_prompt(text: str, words) -> tuple[str, list[str]]:
    keys = ["grammar", "creativity", "consistency"] + (["obedience"] if words else [])
    instruction = (
        f"The student was asked to write a story that uses these words: {', '.join(words)}.\n\n" if words else ""
    )
    criteria = "\n".join(f"- {k}: {RUBRIC[k]}" for k in keys)
    schema = ", ".join(f'"{k}": <1-10>' for k in keys)
    prompt = (
        "You are a teacher grading a short story for young children written by a student. "
        "Be strict and use the whole 1-10 scale (1 = very poor, 10 = excellent). "
        "The story may stop abruptly because of a length limit; do not penalize the missing ending.\n\n"
        f"{instruction}Story:\n<<<\n{text}\n>>>\n\n"
        f"Grade each criterion with an integer from 1 to 10:\n{criteria}\n\n"
        f"Answer with only this JSON object: {{{schema}}}"
    )
    return prompt, keys


def parse_scores(raw: str, keys: list[str]) -> tuple[dict | None, str]:
    """Primer objeto JSON con todas las `keys` como enteros 1-10. Devuelve (notas o None, estado)."""
    decoder = json.JSONDecoder()
    first_brace = raw.find("{")
    i = first_brace
    while i != -1:
        try:
            obj, end = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            i = raw.find("{", i + 1)
            continue
        if isinstance(obj, dict) and all(k in obj for k in keys):
            try:
                scores = {k: int(round(float(obj[k]))) for k in keys}
            except (TypeError, ValueError):
                return None, "valores no numéricos"
            if not all(1 <= v <= 10 for v in scores.values()):
                return None, "fuera de rango"
            extra = raw[:first_brace].strip() or raw[end:].strip()
            return scores, "ok con texto extra" if extra else "ok"
        i = raw.find("{", end if isinstance(obj, dict) else i + 1)
    return None, "sin JSON válido con esas claves" if first_brace != -1 else "sin JSON"


def fake_judge(prompt: str, keys: list[str], seed: int) -> str:
    """Solo SMOKE_TEST: respuestas deterministas, a veces rotas a propósito, para ejercitar el parser."""
    h = int(hashlib.sha1(f"{seed}|{prompt}".encode()).hexdigest(), 16)
    obj = {k: 1 + (h >> (4 * n)) % 10 for n, k in enumerate(keys)}
    kind = h % 6
    if kind == 1:
        return json.dumps(obj) + "\n" + json.dumps(obj)  # dos objetos: la regex glotona rompe acá
    if kind == 2:
        return "Here is my evaluation: " + json.dumps(obj) + " Hope it helps!"
    if kind == 3:
        return json.dumps({k: v for k, v in obj.items() if k != keys[-1]})  # falta una clave
    if kind == 4:
        return "I think the story is fine."
    return json.dumps(obj)


def call_judge(prompt: str, keys: list[str], seed: int = SEED) -> str:
    if not JUDGE_IS_REAL:
        return fake_judge(prompt, keys, seed)
    import ollama

    response = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        think=False,
        options={"temperature": 0, "seed": seed},
    )
    return response["message"]["content"]


# %% [markdown]
r"""
### Correr el juez (con caché)

Cada respuesta se guarda en `juez_respuestas.jsonl` con una clave que depende del juez, del prompt de
evaluación y del texto. Volver a correr la celda no repite llamadas ya hechas; cambiar la rúbrica o
regenerar un texto invalida solo esa entrada. Si el parseo falla, se reintenta una vez **con otra
semilla**: con temperatura 0, la misma semilla devolvería exactamente la misma respuesta rota.
"""

# %%
CACHE_PATH = OUTPUT_DIR / ("juez_respuestas.jsonl" if JUDGE_IS_REAL else "juez_falso_respuestas.jsonl")
cache = {}
if CACHE_PATH.exists():
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            cache[entry["key"]] = entry

t0 = time.perf_counter()
results = []
with open(CACHE_PATH, "a", encoding="utf-8") as cache_file:
    for n, row in enumerate(items.itertuples(index=False)):
        prompt, keys = grading_prompt(row.texto, row.palabras)
        key = hashlib.sha1(f"{JUDGE_MODEL}|{JUDGE_IS_REAL}|{prompt}".encode()).hexdigest()
        entry = cache.get(key)
        if entry is None or entry["scores"] is None:
            raws = []
            scores, status = None, "sin llamar"
            for attempt in range(2):
                try:
                    # Con temperatura 0, repetir la misma semilla repetiría la misma respuesta rota.
                    raw = call_judge(prompt, keys, seed=SEED + attempt)
                except Exception as e:  # un error de red o del servidor no tira abajo la corrida
                    raw = f"<error: {type(e).__name__}: {e}>"
                raws.append(raw)
                scores, status = parse_scores(raw, keys)
                if scores is not None:
                    break
            entry = {"key": key, "scores": scores, "status": status, "attempts": len(raws), "raw": raws}
            cache[key] = entry
            cache_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            cache_file.flush()
        results.append(entry)
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(items)} evaluados · {time.perf_counter() - t0:.0f}s")

judged = items.copy()
judged["estado"] = [e["status"] for e in results]
judged["intentos"] = [e["attempts"] for e in results]
for es, en in CRITERIA.items():
    judged[es] = [e["scores"].get(en, np.nan) if e["scores"] else np.nan for e in results]
judged.drop(columns=["palabras"]).to_csv(OUTPUT_DIR / "notas.csv", index=False)
print(f"listo en {time.perf_counter() - t0:.0f}s")

# %% [markdown]
r"""
### ¿El juez respetó el formato?
"""

# %%
format_table = pd.crosstab(judged["fuente"], judged["estado"], margins=True)
format_table.to_csv(OUTPUT_DIR / "formato_juez.csv")
print(format_table.to_string())
print(f"\nrespuestas que necesitaron reintento: {(judged['intentos'] > 1).mean():.1%}")
bad = judged[judged["gramática"].isna()]
if len(bad):
    print("\nejemplo de respuesta que no se pudo usar:", cache[results[bad.index[0]]["key"]]["raw"][-1][:300])

# %% [markdown]
r"""
## Notas por modelo

Media por fuente y tipo de prompt, con un intervalo bootstrap al 95% que remuestrea **prompts** (las
muestras de un mismo prompt no son independientes). Primero las anclas: si "cuento real" no le gana
claramente a "palabras mezcladas" en gramática, el juez no está leyendo gramática.
"""


# %%
def bootstrap_ci(df: pd.DataFrame, col: str, n_boot=2_000, seed=SEED) -> tuple[float, float, float]:
    per_prompt = df.groupby("prompt")[col].mean().dropna().to_numpy()
    if len(per_prompt) == 0:
        return np.nan, np.nan, np.nan
    boots = np.random.default_rng(seed).choice(per_prompt, size=(n_boot, len(per_prompt))).mean(axis=1)
    return per_prompt.mean(), *np.percentile(boots, [2.5, 97.5])


SOURCES = list(dict.fromkeys(judged["fuente"]))
rows = []
for (source, kind), g in judged.groupby(["fuente", "tipo"], sort=False):
    row = {"fuente": source, "tipo": kind, "n": g["gramática"].notna().sum()}
    for crit in CRITERIA:
        if g[crit].notna().any():
            mean, lo, hi = bootstrap_ci(g, crit)
            row[crit] = mean
            row[f"{crit} IC95"] = f"[{lo:.1f}, {hi:.1f}]"
    row["uso de palabras"] = g["uso de palabras"].mean()
    row["NLL (ref)"] = g["NLL (ref)"].mean()
    rows.append(row)
scores = pd.DataFrame(rows).set_index(["fuente", "tipo"])
scores.to_csv(OUTPUT_DIR / "notas_por_modelo.csv")
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(scores.to_string(float_format="{:.2f}".format))

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), gridspec_kw={"width_ratios": [3, 4]})
for ax, (kind, crits) in zip(axes, (("cuento", list(CRITERIA)[:3]), ("instrucción", list(CRITERIA)))):
    sub = judged[judged["tipo"] == kind]
    sources = [s for s in SOURCES if s in set(sub["fuente"])]
    width = 0.8 / len(sources)
    for k, source in enumerate(sources):
        g = sub[sub["fuente"] == source]
        stats = [bootstrap_ci(g, c) for c in crits]
        means = [s[0] for s in stats]
        err = [[m - s[1] for m, s in zip(means, stats)], [s[2] - m for m, s in zip(means, stats)]]
        hatch = "//" if source.startswith("ancla") else None
        ax.bar(np.arange(len(crits)) + k * width, means, width, yerr=err, capsize=2, label=source, hatch=hatch, alpha=0.85)
    ax.set_xticks(np.arange(len(crits)) + 0.4 - width / 2, crits)
    ax.set(ylim=(0, 10.5), ylabel="nota media (1-10)", title=f"Prompts de {kind}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "notas.png", dpi=110)
plt.show()

# %% [markdown]
r"""
## Juez contra perplejidad

Dos lecturas:

1. **Por modelo:** ¿el orden que da la perplejidad (menor es mejor) es el mismo que da el juez? Con 2-4
   modelos no hay estadística que valga; se lee el orden y el tamaño de las diferencias contra sus IC.
2. **Por texto:** correlación de Spearman entre la NLL de referencia de cada texto y cada nota del juez,
   sobre todas las generaciones de los modelos (sin anclas) y, aparte, incluyendo anclas. Si la NLL mide
   "suena a TinyStories", debería correlacionar negativo con gramática; con creatividad no hay por qué.

Se suman, si están, los números de la Etapa 2 (prompts de recuerdo y consistencia) y la Etapa 4 (uso de
palabras, pérdida en Instruct).
"""

# %%
story_scores = scores.xs("cuento", level="tipo")
model_view = perplexity.join(story_scores[list(CRITERIA)[:3]], how="left")
model_view = model_view.join(scores.xs("instrucción", level="tipo")[["obediencia", "uso de palabras"]], how="left")
model_view["rango ppl"] = model_view["val ppl"].rank()
model_view["rango juez (gram+cons)"] = (-(model_view["gramática"] + model_view["consistencia"])).rank()

stage2_path = CKPT_DIR / "stage2_ablation.csv"
if stage2_path.is_file():
    stage2 = pd.read_csv(stage2_path, index_col=0).T.rename(index={"wide": "base (ancho)", "deep": "profundo"})
    keep = [c for c in stage2.columns if c.startswith(("recuerdo", "consistencia"))]
    model_view = model_view.join(stage2[keep].astype(float).add_prefix("Etapa 2 · "), how="left")
else:
    print(f"aviso: no está {stage2_path.name}; la comparación con la Etapa 2 queda sin sus prompts de prueba.")

stage4_path = CKPT_DIR / "stage4" / "comparacion_lora.csv"
if stage4_path.is_file():
    stage4 = pd.read_csv(stage4_path, index_col=0).drop(columns=["LoRA / completo"], errors="ignore").T
    stage4 = stage4.rename(index={"base": "base (ancho)", "completo": "SFT completo", "lora": "SFT LoRA"})
    model_view = model_view.join(stage4[["pérdida Instruct val"]].astype(float).add_prefix("Etapa 4 · "), how="left")

model_view.to_csv(OUTPUT_DIR / "juez_vs_perplejidad.csv")
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(model_view.T.to_string(float_format="{:.3f}".format))

# %%
model_rows = judged[~judged["fuente"].str.startswith("ancla")]
corr = pd.DataFrame(
    {
        label: {crit: df[["NLL (ref)", crit]].dropna().corr(method="spearman").iloc[0, 1] for crit in CRITERIA}
        for label, df in (("modelos", model_rows), ("modelos + anclas", judged))
    }
)
corr.to_csv(OUTPUT_DIR / "correlacion_nll_juez.csv")
print("Spearman entre NLL de referencia por texto y nota del juez (negativo = coinciden):")
print(corr.to_string(float_format="{:.2f}".format))

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
for ax, crit in zip(axes, list(CRITERIA)[:3]):
    for source in SOURCES:
        g = judged[(judged["fuente"] == source) & (judged["tipo"] == "cuento")]
        jitter = np.random.default_rng(SEED).uniform(-0.2, 0.2, len(g))
        ax.scatter(g["NLL (ref)"], g[crit] + jitter, s=14, alpha=0.7, label=source)
    ax.set(xlabel=f"NLL por token bajo {REFERENCE}", ylabel=crit, ylim=(0, 10.8), title=f"{crit} vs. NLL")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "juez_vs_nll.png", dpi=110)
plt.show()

# %% [markdown]
r"""
### Obediencia del juez contra la cuenta automática de la Etapa 4

La Etapa 4 mide "usa las palabras" con una regex. El juez mide lo mismo leyendo. Si no correlacionan, o el
juez no está mirando la instrucción, o la regex se pierde algo (por ejemplo, una palabra usada con otro
sentido). Leer un par de casos donde discrepan.
"""

# %%
instr = judged[(judged["tipo"] == "instrucción") & judged["obediencia"].notna()]
print(f"Spearman obediencia del juez vs. uso de palabras: {instr[['obediencia', 'uso de palabras']].corr(method='spearman').iloc[0, 1]:.2f}")
disagree = instr.assign(gap=(instr["obediencia"] / 10 - instr["uso de palabras"]).abs()).sort_values("gap", ascending=False)
for _, row in disagree.head(3).iterrows():
    print("-" * 100)
    print(f"{row['fuente']} · uso de palabras {row['uso de palabras']:.2f} · obediencia del juez {row['obediencia']:.0f}")
    print(row["texto"][:400])

with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "smoke_test": SMOKE_TEST,
            "juez": JUDGE_MODEL if JUDGE_IS_REAL else "falso (smoke)",
            "modelos": {name: str(path) for name, path in MODEL_PATHS.items() if path.is_file()},
            "referencia_nll": REFERENCE,
            "n_samples": N_SAMPLES,
            "story_prompts": STORY_PROMPTS,
            "word_prompts": WORD_PROMPTS,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

# %% [markdown]
r"""
## Resultados de la corrida completa

Corrida local (RTX 5070 Laptop, 8 GB), todas las etapas en orden, semilla 1337, juez **qwen3:4b** con
`think=False` y `format="json"`. **264 evaluaciones**: 4 modelos × 48 textos (6 prompts de cuento y 6 de
instrucción × 4 muestras) más 72 anclas — cuentos reales, esos mismos cuentos con las palabras mezcladas, y
cuentos reales evaluados contra una instrucción que no les corresponde. Las cifras salen de
`checkpoints/stage5/`. En la ejecución que quedó guardada en este notebook las respuestas del juez se leyeron
de la caché de la primera corrida completa (`juez_respuestas.jsonl`), y por eso la celda dice "listo en 0s";
esa primera vez, las 264 llamadas tardaron 165 segundos. Los textos evaluados son los mismos: las generaciones
salen idénticas con la misma semilla.

### 1. ¿Es confiable el juez? Sí para gramática y coherencia, no para obediencia

- **Formato: 264 de 264 respuestas válidas** (`formato_juez.csv`), ninguna con texto extra ni con dos
  objetos. Con `format="json"`, la trampa de "el juez contesta dos veces" no apareció en esta corrida; el
  parser robusto quedó como seguro, sin llegar a usarse.
- **Separa bien las anclas** (`notas_por_modelo.csv`):

  | ancla | gramática | creatividad | consistencia |
  |---|---|---|---|
  | cuento real | 8,3 [8,0, 8,5] | 6,8 [6,5, 7,1] | 8,8 [8,6, 9,0] |
  | palabras mezcladas | 1,5 [1,3, 1,8] | 2,5 [2,2, 2,7] | 1,5 [1,3, 1,8] |

  Mismo vocabulario y misma temática, con la gramática destruida: la nota cae de 8,3 a 1,5 y los intervalos
  no se pisan. El juez lee gramática y coherencia, no solo el tema.
- **La obediencia no es confiable.** Los cuentos reales evaluados contra una instrucción ajena (12,5 % de
  uso de las palabras pedidas) sacaron obediencia 5,75 [4,9, 6,6]. Los del SFT completo sacaron 4,0 tanto
  con 0 como con 3 de 3 palabras usadas. La correlación entre la obediencia del juez y el uso real de
  palabras es de apenas 0,23 (Spearman): la obediencia está contaminada por la calidad general del texto
  (efecto halo). Para obediencia, la cuenta automática de la Etapa 4 es el mejor instrumento.

Notas por modelo (prompts de cuento; en obediencia y uso de palabras, prompts de instrucción):

| | val ppl (liso) | gramática | creatividad | consistencia | obediencia | uso de palabras |
|---|---|---|---|---|---|---|
| base (ancho) | **10,3** | 1,96 | 3,08 | 2,17 [2,0, 2,4] | 2,8 [2,3, 3,3] | 18 % |
| profundo | 10,6 | 1,92 | 2,92 | 1,96 [1,9, 2,0] | 2,5 [2,0, 3,0] | 21 % |
| SFT completo | 13,1 | 2,00 | 2,96 | 1,88 | **4,1 [3,9, 4,4]** | **56 %** |
| SFT LoRA | 12,7 | 1,96 | 2,92 | 1,83 | 3,7 [3,2, 4,0] | 33 % |

**Hay un efecto piso.** De las 192 generaciones de nuestros modelos, el juez puso gramática 2 en 173 y 1 en
19, nunca más. Contra la escala de un cuento real (8,3), los cuatro modelos están en el mismo escalón: el
juez no puede ordenarlos en gramática. Las diferencias en consistencia son de décimas y, salvo base contra
el resto, los intervalos se pisan.

### 2. Dónde le da la razón el juez a la perplejidad

- **Texto por texto, dentro de nuestros modelos:** cuanto menor la NLL de un texto (medida con el modelo
  ancho), mejor la nota. Spearman (`correlacion_nll_juez.csv`): gramática −0,35, creatividad −0,37,
  consistencia −0,44; con las anclas incluidas, −0,49, −0,49 y −0,52. El signo es el esperado en las tres,
  incluso en creatividad.
- **Base contra profundo:** la perplejidad prefiere al ancho (10,3 contra 10,6) y el juez también, por poco
  (consistencia 2,17 contra 1,96).
- **Contra la Etapa 2:** esa preferencia coincide con la probabilidad media de la respuesta correcta en los
  prompts de recuerdo (0,12 contra 0,10). No coincide con el reuso de palabras clave, que favorecía al
  profundo (0,58 contra 0,52). Ninguna de las tres lecturas reproduce "profundidad ↔ contexto" del paper a
  esta escala.

### 3. Dónde se le va para otro lado, y qué dice eso de la perplejidad

1. **El SFT tiene la peor perplejidad y el juez no lo castiga.**
   - En texto liso, el SFT completo sube de 10,3 a 13,1 de perplejidad: es el peor de los cuatro.
   - El juez no lo ve peor: gramática igual (2,0) y la obediencia más alta (4,1 contra 2,8).
   - La perplejidad sobre TinyStories mide cuánto se parece el modelo a esa distribución. El SFT se corrió
     hacia otro formato, y eso se paga en perplejidad aunque los cuentos no empeoren.
   - En su propia distribución (Instruct), el SFT baja de 3,20 a 2,12. La perplejidad depende de sobre qué
     texto se mide; la calidad no.
2. **Un cuento real puntúa casi igual que el balbuceo del propio modelo.**
   - Con el modelo ancho como vara, los cuentos reales tienen una NLL de 2,39 por token, y las generaciones
     del propio ancho, 2,29: al modelo le parecen más probables sus propias muestras que los cuentos
     humanos.
   - El juez las separa por más de 6 puntos (gramática 8,3 contra 1,96).
   - La perplejidad bajo un modelo mide cuánto se parece un texto a lo que ese modelo produciría, no si es
     bueno. Por eso el propio modelo se autoevalúa bien.
   - Solo el caso extremo de las palabras mezcladas (NLL 8,8) lo separan las dos varas por igual.

**En una línea:** la perplejidad sirve para detectar texto roto y para comparar modelos sobre la misma
distribución; no sirve para decidir si un texto es un buen cuento, y penaliza cualquier cambio de formato
como si fuera pérdida de calidad. El juez sirve justo para eso, con dos límites medidos: satura contra un
piso cuando todos los modelos son malos, y su obediencia se deja llevar por la calidad general del texto.

### 4. Obediencia: el SFT sube, y acá la regex le gana al juez

- El SFT completo sube la obediencia del juez de 2,8 a 4,1 [3,9, 4,4] sobre los mismos prompts, y el uso de
  palabras de 18 % a 56 %; LoRA queda en el medio (3,7 y 33 %). En la dirección grande, las dos medidas
  coinciden.
- Texto por texto dejan de coincidir: Spearman 0,23. Los tres casos de mayor discrepancia son anclas de
  instrucción ajena — cuentos reales, bien escritos, con 0 de las palabras pedidas — a los que el juez puso
  obediencia 9, 7 y 7. Ahí la razón es la regex, no el juez: cuenta lo que dice contar, mientras que el
  juez mezcla obediencia con calidad.
- Por eso la conclusión de la Etapa 4 se apoya en `uso − piso` y no en la nota del juez.

### Limitaciones

- 24 textos por modelo y tipo, y una sola semilla de entrenamiento por configuración. Alcanza para las
  diferencias grandes (anclas, obediencia del SFT), no para las de décimas.
- El juez tiene 4B parámetros. Con un juez más grande, el efecto piso probablemente se abriría en más
  escalones.
"""

# %% [markdown]
r"""
## Cierre del proyecto: qué muestra el pipeline completo

**La tesis de la consigna es "misma arquitectura, mejores datos → aparece el significado". Nuestra lectura, con
nuestros números: aparece, pero en un orden. Primero la forma, después —y a medias— el contenido.** Lo vimos tres
veces, en tres etapas distintas, y coincide con lo que reporta el paper (la gramática llega antes que la
consistencia):

1. **Preentrenamiento (Etapa 2).** El mismo GPT de la clase pasa de adivinar al azar (pérdida 9,06 ≈ ln 8.192) a sus
   primeras oraciones bien formadas en unos 500 pasos (pérdida 3,3), sostiene un personaje con nombre desde el paso
   2.000 (2,56) y termina en perplejidad 10,1. La coherencia del cuento entero, en cambio, no llega: el juez le pone
   consistencia 2,2 sobre 10, contra 8,8 de un cuento real.
2. **SFT (Etapa 4).** En 1.500 pasos (1,7 minutos) el modelo aprende el formato —pasa de 0 % a 92,5 % de cuentos
   cuando solo le damos la línea `Words:`—, mientras que usar las palabras pedidas se mueve mucho menos: la ganancia
   sobre el azar pasa de 0,11 a 0,32, y con la plantilla `solo Words` no cambia.
3. **Evaluación (Etapa 5).** La perplejidad mide forma: cuánto se parece un texto al corpus. Por eso castiga al SFT
   por cambiar de formato (10,3 → 13,1) aunque el juez no lo vea peor, y por eso el modelo encuentra más probables
   sus propias muestras (NLL 2,29) que los cuentos reales (2,39), que el juez separa por 6 puntos de gramática.

**Cómo se encadenan las piezas.** El tokenizador decide las unidades: con 8.192 entradas, casi cada palabra es un
token (Etapa 1). El preentrenamiento les da relaciones: *dog* termina junto a *cat* y *puppy*, y *spoon* junto a
*fork* y *bowl*, muy por encima de los controles al azar (Etapa 3); y además aprende la gramática local. El SFT no
agrega conocimiento: reusa esas representaciones para instalar un comportamiento, y lo que más le cuesta es
justo lo que pide contenido, como usar una palabra 50 a 100 tokens después de haberla leído. El juez, por último,
muestra lo que la perplejidad no ve. Cada etapa se apoya en la anterior: los vecinos de la Etapa 3 existen porque
el tokenizador hizo de *dog* una sola fila, y el SFT funciona tan rápido porque el base ya sabía escribir cuentos
(con `Words + Story` arranca uno el 92,5 % de las veces antes de cualquier SFT).

**Lo que no probamos, y habría que probar.** Nuestro contraste con "datos peores" es el del paper y el del
Transformer de caracteres sobre Shakespeare que vimos en clase, y ese contraste no mueve una sola variable: cambian
los datos, pero también el tokenizador (caracteres contra BPE) y la escala. El experimento limpio es entrenar este
mismo modelo, con los mismos pasos, sobre 82M tokens de texto web (lo planteamos en la Etapa 0). Con más tiempo,
eso iría primero; después, tres semillas para la ablación y un SFT que enmascare la instrucción.

El registro de decisiones del proyecto —con qué contábamos, qué recortamos y por qué, y dónde terminamos contra lo
planeado— está al final del notebook `00_dataset`, junto con cómo usamos IA.
"""
