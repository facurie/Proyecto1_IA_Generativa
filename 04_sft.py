# %% [markdown]
r"""
# Etapa 4 · SFT — comportamiento, no conocimiento

Seguimos entrenando el modelo **ancho** de la Etapa 2 con **la misma pérdida de próximo token**; lo único
que cambia son los datos: en lugar de cuentos sueltos, bloques de TinyStories-Instruct con una instrucción
(`Features:` / `Words:` / `Summary:` / `Random sentence:`, un subconjunto al azar en orden al azar), después
`Story:` y el cuento. Es formato de continuación pelado: no hay chat, ni turnos, ni máscara sobre la
instrucción.

Tres modelos, mismos prompts y mismas semillas:

- **base**: `pretrain_wide.pt` tal cual salió de la Etapa 2.
- **SFT completo**: todos los parámetros se entrenan sobre Instruct.
- **SFT LoRA** (opcional de la consigna): pesos base congelados; solo se entrenan adaptadores de rango bajo
  escritos a mano sobre `Head.key/query/value` y `MultiHeadAttention.proj`.

El comportamiento se mide en **dos ejes por separado**, porque se pueden mover por su cuenta:

- **Forma** (*cómo* escribe): ¿cuando le piden un cuento, arranca un cuento y lo termina?
- **Contenido** (*qué* escribe): ¿usa las palabras que le dieron, más de lo que las usaría por azar?

**Lee:** `checkpoints/tokenizer.json` (Etapa 1), `checkpoints/pretrain_wide.pt` (Etapa 2).
**Produce:** `checkpoints/sft_final.pt` (SFT completo, lo lee la Etapa 5), `checkpoints/sft_lora.pt`
(LoRA ya fusionado en los pesos, se carga igual que cualquier otro checkpoint), y tablas/gráficos en
`checkpoints/stage4/`.

**Cómo correrlo.** En CPU, `SMOKE_TEST` se activa solo: usa `checkpoints/smoke/pretrain_wide.pt`, un puñado
de registros del split `validation` de Instruct y ~100 pasos, así no hace falta materializar el split
`train` entero (~21,7M de líneas, ver `data.py`). En la T4 hace la corrida completa. Si un checkpoint de
SFT ya existe con la misma configuración, se carga sin reentrenar.
"""

# %%
import environment  # noqa: F401  (primero siempre: caché de HF, stdout UTF-8, checkpoints/)

environment.summary()

import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.nn import functional as F

from data import load_instruct_records, load_tinystories

DEVICE = environment.device()
SMOKE_TEST = DEVICE == "cpu"  # CPU: validar el pipeline. GPU: la corrida de verdad. Se puede forzar a mano.
SEED = 1337

CKPT_DIR = environment.CHECKPOINTS / "smoke" if SMOKE_TEST else environment.CHECKPOINTS
TOKENIZER_PATH = environment.CHECKPOINTS / "tokenizer.json"
BASE_PATH = CKPT_DIR / "pretrain_wide.pt"
SFT_PATH = CKPT_DIR / "sft_final.pt"
LORA_PATH = CKPT_DIR / "sft_lora.pt"
OUTPUT_DIR = CKPT_DIR / "stage4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

USE_AMP = DEVICE == "cuda"

torch.manual_seed(SEED)
print(f"dispositivo: {DEVICE} · SMOKE_TEST: {SMOKE_TEST} · AMP: {USE_AMP} · checkpoints en: {CKPT_DIR}")

# %% [markdown]
r"""
## El modelo de la Etapa 2

Las clases son copia textual de `02_pretraining.py` (los nombres de módulos tienen que coincidir para que
el `state_dict` cargue sin adivinar). No se importan de ahí porque importar ese archivo lo ejecutaría
entero, entrenamiento incluido.
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
        # ignore_index=-100 es el default de cross_entropy: así se ignora el relleno de los batches de SFT.
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
for path, stage in ((TOKENIZER_PATH, "la Etapa 1 (01_tokenizer)"), (BASE_PATH, "la Etapa 2 (02_pretraining)")):
    if not path.is_file():
        raise FileNotFoundError(f"No está {path}. Corré primero {stage} en esta misma sesión, o traé el archivo de Drive.")

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
VOCAB_SIZE = tokenizer.get_vocab_size()
EOT_ID = tokenizer.token_to_id("<|endoftext|>")

base_model, base_ckpt = load_model(BASE_PATH)
MODEL_CFG = base_model.cfg
BLOCK_SIZE = MODEL_CFG.block_size
assert MODEL_CFG.vocab_size == VOCAB_SIZE, "el checkpoint base y el tokenizer no son del mismo vocabulario"
base_hist = base_ckpt["history"][-1] if base_ckpt["history"] else {}
print(f"base: {BASE_PATH.name} · paso {base_ckpt['step']:,} · val loss (TinyStories) {base_hist.get('val_loss', float('nan')):.3f}")
print(f"forma: {asdict(MODEL_CFG)}")

# %% [markdown]
r"""
## Datos: bloques instrucción → cuento

`load_instruct_records` reagrupa las líneas de TinyStories-Instruct en registros completos. Cada ejemplo de
entrenamiento es **un registro**, alineado al principio: `<|endoftext|>` + registro + `<|endoftext|>`.

A diferencia de la Etapa 2, acá **no** se cortan ventanas al azar de un arreglo concatenado: una ventana
que arranca en el medio de un cuento pierde la instrucción, y entonces el modelo ve el cuento sin la
consigna que lo explica, que es justo la asociación que queremos enseñar. Si el registro no entra en
`block_size + 1` tokens, se trunca **el final del cuento** (la instrucción queda siempre). Los lugares de
relleno llevan `-100` como objetivo, que `cross_entropy` ignora. La pérdida es la misma de siempre sobre
todo el texto, instrucción incluida: no enmascaramos la instrucción.

Split held-out: el `validation` de Instruct, que no se toca para entrenar. En `SMOKE_TEST` sale todo de
`validation` (partes disjuntas) para no materializar `train`.
"""

# %%
N_TRAIN_RECORDS = 2_000 if SMOKE_TEST else 100_000
N_VAL_RECORDS = 200 if SMOKE_TEST else 2_000

t0 = time.perf_counter()
if SMOKE_TEST:
    smoke_records = load_instruct_records("validation", limit=N_TRAIN_RECORDS + N_VAL_RECORDS)
    train_records, val_records = smoke_records[:N_TRAIN_RECORDS], smoke_records[N_TRAIN_RECORDS:]
else:
    train_records = load_instruct_records("train", limit=N_TRAIN_RECORDS)
    val_records = load_instruct_records("validation", limit=N_VAL_RECORDS)
print(f"registros: train {len(train_records):,} · val {len(val_records):,} · cargados en {time.perf_counter() - t0:.0f}s")

HEADER_FIELDS = ("Features", "Words", "Summary", "Random sentence")
header_counts = Counter(f for r in train_records for f in HEADER_FIELDS if re.search(rf"^{f}:", r, flags=re.M))
print("fracción de registros con cada encabezado:", {f: f"{header_counts[f] / len(train_records):.0%}" for f in HEADER_FIELDS})

# Cómo se escribe exactamente el separador (¿"Story:" con espacio? ¿una línea en blanco?): se mide en los
# datos en vez de suponerlo, porque el prompt de evaluación tiene que calcar el formato de entrenamiento.
story_markers = Counter(m.group(0) for r in train_records if (m := re.search(r"Story:[^\n]*\n\n?", r)))
STORY_MARKER = story_markers.most_common(1)[0][0]
print(f"separador más común: {STORY_MARKER!r} ({story_markers.most_common(1)[0][1] / len(train_records):.0%} de los registros)")
print("\n--- un registro de ejemplo ---")
print(train_records[0])


# %%
def encode_records(records: list[str]) -> list[np.ndarray]:
    """Cada registro como `<|endoftext|>` + ids + `<|endoftext|>`, en `uint16` como en la Etapa 2."""
    assert VOCAB_SIZE <= np.iinfo(np.uint16).max
    return [np.array([EOT_ID] + e.ids + [EOT_ID], dtype=np.uint16) for e in tokenizer.encode_batch(records)]


train_examples = encode_records(train_records)
val_examples = encode_records(val_records)
lengths = np.array([len(e) for e in train_examples])
print(f"tokens por registro: mediana {np.median(lengths):.0f} · p90 {np.percentile(lengths, 90):.0f} · máx {lengths.max()}")
print(f"registros que se truncan con block_size = {BLOCK_SIZE}: {(lengths > BLOCK_SIZE + 1).mean():.1%}")
print(f"tokens de SFT disponibles: {lengths.sum() / 1e6:.2f}M")


def get_sft_batch(examples: list[np.ndarray], batch_size: int, generator: torch.Generator):
    """Registros al azar, alineados al principio; relleno con EOT en x y con -100 (ignorado) en y."""
    picks = torch.randint(len(examples), (batch_size,), generator=generator).tolist()
    x = torch.full((batch_size, BLOCK_SIZE), EOT_ID, dtype=torch.long)
    y = torch.full((batch_size, BLOCK_SIZE), -100, dtype=torch.long)
    for row, i in enumerate(picks):
        ids = torch.from_numpy(examples[i][: BLOCK_SIZE + 1].astype(np.int64))
        x[row, : len(ids) - 1] = ids[:-1]
        y[row, : len(ids) - 1] = ids[1:]
    return x.to(DEVICE), y.to(DEVICE)


# %% [markdown]
r"""
## LoRA a mano

`LoRALayer` y `LinearWithLoRA` son las de la consigna. `add_lora` envuelve, en cada bloque, las
proyecciones `key`/`query`/`value` de cada cabeza y la `proj` de salida de la atención, y congela todo lo
demás. `merge_lora` hace la vuelta: suma `ΔW = scaling · (A·B)ᵀ` al peso congelado y devuelve un
`GPTLanguageModel` común, así el checkpoint de LoRA se carga igual que cualquier otro (y la Etapa 5 no
necesita saber que existió LoRA). Se chequea que el modelo fusionado dé los mismos logits que el envuelto.
"""


# %%
class LoRALayer(nn.Module):
    """Aprende ΔW ≈ (alpha/r) · B · A, con A al azar y chica y B arrancando en cero, así ΔW=0 al principio."""

    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std = 1.0 / math.sqrt(rank)
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.scaling * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    """Envuelve un nn.Linear que ya existe: congela `linear` y entrena solo la LoRALayer."""

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False
        # El `.to()` importa: el modelo se envuelve DESPUÉS de mandarlo a la GPU y un nn.Parameter nuevo nace
        # en la CPU. Sin esto, el primer forward en GPU tira "Expected all tensors to be on the same device".
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha).to(
            device=linear.weight.device, dtype=linear.weight.dtype
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def add_lora(model: GPTLanguageModel, rank: int, alpha: float) -> GPTLanguageModel:
    """Congela todo el modelo y envuelve con LoRA las proyecciones de atención de cada bloque."""
    for p in model.parameters():
        p.requires_grad = False
    for block in model.blocks:
        for head in block.sa.heads:
            head.key = LinearWithLoRA(head.key, rank, alpha)
            head.query = LinearWithLoRA(head.query, rank, alpha)
            head.value = LinearWithLoRA(head.value, rank, alpha)
        block.sa.proj = LinearWithLoRA(block.sa.proj, rank, alpha)
    return model


@torch.no_grad()
def merge_lora(model: GPTLanguageModel) -> GPTLanguageModel:
    """Devuelve una copia sin envoltorios, con ΔW sumado a cada peso. `x @ A @ B` equivale a `x @ (A·B)`, y
    nn.Linear guarda W con forma (out, in), así que ΔW = scaling · (A·B)ᵀ."""
    merged = GPTLanguageModel(model.cfg).to(DEVICE)
    state = {}
    for name, tensor in model.state_dict().items():
        if ".lora." in name:
            continue
        state[name.replace(".linear.", ".")] = tensor.clone()
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            delta = module.lora.scaling * (module.lora.A @ module.lora.B).T
            state[f"{name}.weight"] += delta.to(state[f"{name}.weight"].dtype)
    merged.load_state_dict(state)
    return merged.eval()


def param_counts(model: nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"entrenables": trainable, "total": total}


# %% [markdown]
r"""
## Cómo medimos forma y contenido

Los prompts calcan el formato de entrenamiento con una sola línea `Words:`. Se prueban dos variantes,
porque cada una aísla una parte distinta del comportamiento:

- **`Words + Story`**: `Words: dragon, happy, forest` + el separador `Story:`. El modelo solo tiene que
  escribir el cuento. Mide sobre todo **contenido**.
- **`solo Words`**: la línea `Words:` y nada más, como en el ejemplo de la consigna. Para escribir un cuento
  el modelo tiene que *saber* que después de la instrucción viene `Story:` y un cuento. Mide sobre todo
  **forma**.

Por generación se registra:

- **arranca cuento** (forma): en `solo Words`, que el modelo mismo escriba `Story:` y después texto
  narrativo; en `Words + Story`, que empiece directo con prosa (una letra) y no con otra línea de encabezado.
- **termina** (forma): que cierre con `<|endoftext|>` dentro del presupuesto de tokens, es decir, que haya
  aprendido dónde termina un cuento.
- **uso de palabras** (contenido): fracción de las palabras pedidas que aparecen en el cuento (con
  flexiones con sufijos explícitos: `dragon` cuenta `dragons`, `happy` cuenta `happily`, pero `cat` no
  cuenta `catch`).
- **piso de azar** (contenido): la misma cuenta, pero con las palabras de *otro* prompt del set. Es el uso
  de palabras que se espera sin ninguna obediencia; "usa las palabras" solo cuenta si le gana a esto.

Las palabras se eligieron antes de correr, de uso frecuente en cuentos infantiles y como un solo token.
Todos los modelos ven los mismos prompts con las mismas semillas.

**Trampa medida: el prompt no puede terminar en espacio o salto de línea.** El pre-tokenizador ByteLevel
agrupa el whitespace distinto según qué venga después: `"Story: \n\nOnce"` en el medio de un registro no
se parte en los mismos tokens que `"Story: \n\n"` al final de un prompt. El modelo recibe entonces una
secuencia de tokens que nunca vio en entrenamiento y responde con basura de bytes sueltos (`�ܲ�Once
upon...`), lo que parece un modelo roto y es un prompt mal cortado. Por eso los prompts terminan en
`Story:` y en la última palabra pedida, y los blancos los escribe el modelo.
"""

# %%
WORD_PROMPTS = [
    ("dragon", "happy", "forest"),
    ("ball", "sad", "garden"),
    ("cat", "brave", "river"),
    ("cake", "angry", "school"),
    ("bird", "scared", "tree"),
    ("boat", "kind", "sea"),
    ("dog", "funny", "park"),
    ("flower", "shy", "rain"),
]
# Los prompts terminan en un carácter no blanco a propósito (ver la trampa del borde, arriba).
TEMPLATES = {
    "Words + Story": lambda words: f"Words: {', '.join(words)}\n{STORY_MARKER.rstrip()}",
    "solo Words": lambda words: f"Words: {', '.join(words)}",
}
N_SAMPLES = 2 if SMOKE_TEST else 5
MAX_NEW_TOKENS = 250
HEADER_RE = re.compile(r"^(Features|Words|Summary|Random sentence|Story):", flags=re.M)

for words in WORD_PROMPTS:
    split = [w for w in words if tokenizer.token_to_id("Ġ" + w) is None]
    if split:
        print(f"aviso: {split} no es un solo token de palabra entera en este vocabulario")


def autocast():
    return torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP)


def generate_ids(model, prompt: str, *, max_new_tokens=MAX_NEW_TOKENS, seed=SEED) -> tuple[str, bool]:
    """Continuación de `[EOT] + prompt` (sin el prompt) y si terminó con `<|endoftext|>`."""
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    idx = torch.tensor([[EOT_ID] + tokenizer.encode(prompt).ids], device=DEVICE)
    with autocast():
        out = model.generate(idx, max_new_tokens, generator=generator, stop_token=EOT_ID)
    model.train(was_training)
    new = out[0, idx.shape[1] :].tolist()
    ended = EOT_ID in new
    if ended:
        new = new[: new.index(EOT_ID)]
    return tokenizer.decode(new), ended


def word_used(word: str, text: str) -> bool:
    """
    La palabra o una flexión con sufijo explícito: `dragon` → dragons; `happy` → happily/happier/happiness.
    Sufijos y no prefijo pelado, porque `cat` no tiene que contar `catch`, ni `happy` contar `happen`.
    """
    forms = rf"{re.escape(word)}(s|es|d|ed|ing|er|est|ly|ness)?"
    if word.endswith("y"):
        forms += rf"|{re.escape(word[:-1])}i(es|ed|er|est|ly|ness)"
    return re.search(rf"\b(?:{forms})\b", text, flags=re.IGNORECASE) is not None


def story_part(template: str, text: str) -> tuple[str, bool]:
    """Separa el cuento de lo que haya escrito antes, y decide si "arrancó un cuento"."""
    if template == "solo Words":
        m = re.search(r"Story:[^\n]*\n*", text)
        if not m:
            return text, False
        story = text[m.end() :]
    else:
        story = text.lstrip()
    starts_story = bool(re.match(r"\s*[A-Za-z\"']", story)) and not HEADER_RE.match(story.lstrip())
    return story, starts_story and len(story.split()) >= 10


def evaluate_behavior(model, n_samples=N_SAMPLES, prompts=WORD_PROMPTS, max_new_tokens=MAX_NEW_TOKENS) -> pd.DataFrame:
    rows = []
    for t, (template, make_prompt) in enumerate(TEMPLATES.items()):
        for i, words in enumerate(prompts):
            control = prompts[(i + len(prompts) // 2) % len(prompts)]  # las palabras de otro prompt
            for j in range(n_samples):
                text, ended = generate_ids(model, make_prompt(words), max_new_tokens=max_new_tokens, seed=SEED + 100 * i + j)
                story, starts = story_part(template, text)
                rows.append(
                    {
                        "plantilla": template,
                        "prompt": i,
                        "muestra": j,
                        "palabras": ", ".join(words),
                        "arranca cuento": starts,
                        "termina": ended,
                        "uso de palabras": np.mean([word_used(w, story) for w in words]),
                        "piso de azar": np.mean([word_used(w, story) for w in control]),
                        "encabezados filtrados": len(HEADER_RE.findall(story)),
                        "largo (palabras)": len(story.split()),
                        "texto": text,
                    }
                )
    return pd.DataFrame(rows)


def summarize_behavior(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["arranca cuento", "termina", "uso de palabras", "piso de azar", "largo (palabras)"]
    return df.groupby("plantilla", sort=False)[cols].mean()


# %% [markdown]
r"""
## Entrenamiento

Mismo esqueleto que la Etapa 2 (AdamW, warmup + coseno, clipping, mixed precision), con dos cambios de la
guía de tamaños: **`lr` más bajo** (1e-4) y **menos pasos**, porque estamos empujando un modelo ya
entrenado, no arrancando de cero. LoRA usa un `lr` más alto (1e-3): arranca en ΔW = 0 y tiene muchos
menos parámetros para moverse, el valor habitual en la literatura de LoRA.

Además de la pérdida en Instruct, en cada evaluación se mide la pérdida sobre **TinyStories liso**
(validation, lo que el base conoce): si sube mucho, el SFT está olvidando cómo escribir en general, no
solo aprendiendo el formato. Y en unos pocos pasos se corre una versión chica de la medición de forma y
contenido, para ver **cuál de los dos ejes se mueve primero**.
"""


# %%
@dataclass
class SFTConfig:
    max_steps: int
    batch_size: int
    eval_interval: int
    eval_iters: int
    lr: float
    min_lr: float
    warmup_steps: int
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    lora_rank: int = 0  # 0 = fine-tuning completo
    lora_alpha: float = 0.0


if SMOKE_TEST:
    common = dict(max_steps=100, batch_size=8, eval_interval=25, eval_iters=5, warmup_steps=10)
else:
    common = dict(max_steps=1_500, batch_size=64, eval_interval=150, eval_iters=40, warmup_steps=50)

FULL = SFTConfig(**common, lr=1e-4, min_lr=1e-5)
LORA = SFTConfig(**common, lr=1e-3, min_lr=1e-4, lora_rank=8, lora_alpha=16)
PROBE_SAMPLES, PROBE_PROMPTS, PROBE_TOKENS = 1, WORD_PROMPTS[:4], 150

tokens_per_run = FULL.max_steps * FULL.batch_size * BLOCK_SIZE
print(FULL)
print(LORA)
print(f"posiciones por corrida (con relleno): {tokens_per_run / 1e6:.1f}M · ≈ {FULL.max_steps * FULL.batch_size / len(train_examples):.2f} épocas de registros")

# El texto liso para medir olvido: mismo formato que en la Etapa 2 (EOT + cuento), alineado al principio.
plain_val_examples = encode_records(list(load_tinystories("validation", limit=N_VAL_RECORDS)["text"]))


def lr_at(step: int, cfg: SFTConfig) -> float:
    """Warmup lineal hasta `lr` y después decaimiento coseno hasta `min_lr`."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, cfg: SFTConfig) -> dict[str, float]:
    """Pérdida media sobre los mismos `eval_iters` batches fijos de cada split."""
    was_training = model.training
    model.eval()
    losses = {}
    for split, examples in (("instruct_train", train_examples), ("instruct_val", val_examples), ("plain_val", plain_val_examples)):
        generator = torch.Generator().manual_seed(SEED)
        batch_losses = []
        for _ in range(cfg.eval_iters):
            x, y = get_sft_batch(examples, cfg.batch_size, generator)
            with autocast():
                _, loss = model(x, y)
            batch_losses.append(loss.item())
        losses[split] = float(np.mean(batch_losses))
    model.train(was_training)
    return losses


def probe(model) -> dict[str, float]:
    """La medición de forma/contenido en chico, para seguirla durante el entrenamiento."""
    df = evaluate_behavior(model, n_samples=PROBE_SAMPLES, prompts=PROBE_PROMPTS, max_new_tokens=PROBE_TOKENS)
    out = {}
    for template, g in df.groupby("plantilla", sort=False):
        key = "sw" if template == "solo Words" else "ws"
        # float() explícito: un escalar de numpy en el historial hace que `torch.load` (weights_only=True,
        # el default desde torch 2.6) se niegue a abrir el checkpoint, acá y en la Etapa 5.
        out[f"{key}_arranca"] = float(g["arranca cuento"].mean())
        out[f"{key}_uso"] = float(g["uso de palabras"].mean())
    return out


def save_checkpoint(path, model, *, step, cfg, history, extra):
    """Mismo formato que la Etapa 2 (model_config + state_dict + historial), escritura atómica."""
    checkpoint = {
        "model_config": asdict(model.cfg),
        "state_dict": model.state_dict(),
        "step": step,
        "train_config": asdict(cfg),
        "history": history,
        "samples": {},
        "base_checkpoint": str(BASE_PATH.name),
        **extra,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)


def finetune(name: str, cfg: SFTConfig, path) -> tuple[GPTLanguageModel, list[dict], dict]:
    """
    Carga el base y lo sigue entrenando sobre Instruct (todo, o solo LoRA si `cfg.lora_rank > 0`).
    Si `path` ya tiene una corrida completa con esta configuración, la carga sin reentrenar.
    Devuelve el modelo listo para generar (con LoRA ya fusionado), el historial y las métricas de costo.
    """
    if path.exists():
        ckpt = torch.load(path, map_location=DEVICE)
        if ckpt["train_config"] == asdict(cfg) and ckpt["step"] >= cfg.max_steps:
            print(f"[{name}] ya estaba entrenado: se carga {path.name} sin reentrenar.")
            model = GPTLanguageModel(GPTConfig(**ckpt["model_config"])).to(DEVICE)
            model.load_state_dict(ckpt["state_dict"])
            return model.eval(), ckpt["history"], ckpt["cost"]
        print(f"[{name}] {path.name} es de otra configuración: se reentrena y se pisa.")

    torch.manual_seed(SEED)  # misma inicialización de A en LoRA y mismo dropout de arranque en cada corrida
    model, _ = load_model(BASE_PATH)
    if cfg.lora_rank:
        model = add_lora(model, cfg.lora_rank, cfg.lora_alpha)
    trainable = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in trainable if p.dim() >= 2]
    no_decay = [p for p in trainable if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr
    )
    scaler = torch.amp.GradScaler(DEVICE, enabled=USE_AMP)
    batch_rng = torch.Generator().manual_seed(SEED)  # el mismo orden de batches para completo y LoRA
    counts = param_counts(model)
    print(f"[{name}] parámetros entrenables: {counts['entrenables']:,} de {counts['total']:,} ({counts['entrenables'] / counts['total']:.2%})")

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    # Lo que ya ocupa la GPU antes del primer paso: este modelo, el base y, en la corrida de LoRA, también el
    # de SFT completo. Se resta del pico para comparar solo lo que agrega entrenar, sin ese sesgo.
    resident = torch.cuda.memory_allocated() if DEVICE == "cuda" else 0
    history = []
    train_time = 0.0
    model.train()
    for step in range(cfg.max_steps + 1):
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            losses = estimate_loss(model, cfg)
            probes = probe(model) if step in (0, cfg.max_steps) or step % (2 * cfg.eval_interval) == 0 else {}
            history.append({"step": step, **losses, **probes, "lr": lr_at(step, cfg), "train_s": train_time})
            probe_txt = f" | uso(W+S) {probes['ws_uso']:.2f} | arranca(sW) {probes['sw_arranca']:.2f}" if probes else ""
            print(
                f"[{name}] paso {step:>5,} | instruct val {losses['instruct_val']:.3f} "
                f"| liso val {losses['plain_val']:.3f}{probe_txt} | {train_time / 60:4.1f} min"
            )
        if step == cfg.max_steps:
            break
        t0 = time.perf_counter()
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)
        x, y = get_sft_batch(train_examples, cfg.batch_size, batch_rng)
        with autocast():
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        train_time += time.perf_counter() - t0

    # Memoria: el pico medido en la GPU por encima de lo residente (activaciones + gradientes + AdamW), y la
    # cuenta analítica de la parte que depende de qué se entrena: gradientes (1 por parámetro entrenable) + los
    # dos momentos de AdamW, en float32. Si el pico casi no cambia entre completo y LoRA, dominan las activaciones.
    cost = {
        **counts,
        "fracción entrenable": counts["entrenables"] / counts["total"],
        "grad + AdamW (MB, analítico)": 3 * counts["entrenables"] * 4 / 1e6,
        "memoria pico GPU al entrenar (MB)": (torch.cuda.max_memory_allocated() - resident) / 1e6 if DEVICE == "cuda" else float("nan"),
        "segundos por paso": train_time / cfg.max_steps,
    }

    if cfg.lora_rank:
        wrapped = model.eval()
        model = merge_lora(wrapped)
        x, _ = get_sft_batch(val_examples, 4, torch.Generator().manual_seed(SEED))
        with torch.no_grad():
            diff = (wrapped(x)[0].float() - model(x)[0].float()).abs().max().item()
        print(f"[{name}] LoRA fusionado · máx |Δlogits| envuelto vs fusionado = {diff:.2e}")
        assert diff < 1e-3, "el modelo fusionado no reproduce al envuelto: revisar merge_lora"

    save_checkpoint(path, model, step=cfg.max_steps, cfg=cfg, history=history, extra={"cost": cost})
    return model.eval(), history, cost


# %% [markdown]
r"""
### SFT completo
"""

# %%
sft_model, sft_history, sft_cost = finetune("completo", FULL, SFT_PATH)

# %% [markdown]
r"""
### SFT con LoRA (rango 8 sobre las proyecciones de atención)
"""

# %%
lora_model, lora_history, lora_cost = finetune("lora", LORA, LORA_PATH)

# %% [markdown]
r"""
## Curvas

Izquierda: la pérdida en Instruct held-out (lo que el SFT optimiza). Centro: la pérdida en TinyStories
liso (lo que el base ya sabía; si sube, hay olvido). Derecha: los dos ejes de comportamiento en la
medición chica, a lo largo de los pasos — **¿se mueve primero la forma o el contenido?**
"""

# %%
HISTORIES = {"completo": sft_history, "lora": lora_history}
COLORS = {"base": "0.5", "completo": "C0", "lora": "C2"}

fig, (ax_i, ax_p, ax_b) = plt.subplots(1, 3, figsize=(16, 4))
for name, hist in HISTORIES.items():
    steps = [h["step"] for h in hist]
    ax_i.plot(steps, [h["instruct_val"] for h in hist], color=COLORS[name], label=name)
    ax_p.plot(steps, [h["plain_val"] for h in hist], color=COLORS[name], label=name)
    probed = [h for h in hist if "ws_uso" in h]
    ax_b.plot([h["step"] for h in probed], [h["ws_uso"] for h in probed], color=COLORS[name], marker="o", label=f"{name} · uso de palabras (W+S)")
    ax_b.plot([h["step"] for h in probed], [h["sw_arranca"] for h in probed], color=COLORS[name], marker="s", ls="--", label=f"{name} · arranca cuento (solo W)")
ax_i.set(xlabel="paso", ylabel="entropía cruzada", title="Instruct (validation)")
ax_p.set(xlabel="paso", ylabel="entropía cruzada", title="TinyStories liso (validation): ¿olvido?")
ax_b.set(xlabel="paso", ylabel="fracción", title="Forma vs. contenido (medición chica)", ylim=(-0.05, 1.05))
for ax in (ax_i, ax_p, ax_b):
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "curvas_sft.png", dpi=110)
plt.show()

# %% [markdown]
r"""
## El experimento que importa: el mismo prompt, antes y después

Primero el par que pide la consigna, `Words: dragon, happy, forest`, en las dos plantillas y con la misma
semilla para los tres modelos. Después, la medición completa sobre los ocho prompts.
"""

# %%
MODELS = {"base": base_model, "completo": sft_model, "lora": lora_model}

for template, make_prompt in TEMPLATES.items():
    prompt = make_prompt(WORD_PROMPTS[0])
    print("=" * 100)
    print(f"PLANTILLA {template!r} · prompt: {prompt!r}")
    for name, model in MODELS.items():
        text, ended = generate_ids(model, prompt, seed=SEED)
        story, _ = story_part(template, text)
        used = [w for w in WORD_PROMPTS[0] if word_used(w, story)]
        print(f"\n--- {name} · usa {used or 'ninguna'} · {'termina' if ended else 'NO termina'} ---")
        print(text.strip())
    print()

# %%
behavior = {name: evaluate_behavior(model) for name, model in MODELS.items()}
all_behavior = pd.concat([df.assign(modelo=name) for name, df in behavior.items()])
all_behavior.to_csv(OUTPUT_DIR / "generaciones.csv", index=False)

summary = pd.concat({name: summarize_behavior(df) for name, df in behavior.items()}, names=["modelo"])
summary["uso − piso"] = summary["uso de palabras"] - summary["piso de azar"]
summary.to_csv(OUTPUT_DIR / "forma_contenido.csv")
print(summary.to_string(float_format="{:.2f}".format))

# %% [markdown]
r"""
### ¿La diferencia de contenido le gana al ruido?

Con pocos prompts, una diferencia de uso de palabras puede ser suerte. Intervalo bootstrap al 95%
remuestreando **prompts** (no generaciones: las muestras del mismo prompt no son independientes) sobre la
diferencia *uso − piso de azar*, por modelo, en la plantilla `Words + Story`.
"""


# %%
def bootstrap_ci(df: pd.DataFrame, n_boot=2_000, seed=SEED) -> tuple[float, float, float]:
    per_prompt = df.groupby("prompt")[["uso de palabras", "piso de azar"]].mean()
    gap = (per_prompt["uso de palabras"] - per_prompt["piso de azar"]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(gap, size=(n_boot, len(gap)), replace=True).mean(axis=1)
    return gap.mean(), *np.percentile(boots, [2.5, 97.5])


ci_rows = {}
for name, df in behavior.items():
    mean, lo, hi = bootstrap_ci(df[df["plantilla"] == "Words + Story"])
    ci_rows[name] = {"uso − piso (media)": mean, "IC95 inf": lo, "IC95 sup": hi, "le gana al azar": lo > 0}
ci_table = pd.DataFrame(ci_rows).T
ci_table.to_csv(OUTPUT_DIR / "contenido_bootstrap.csv")
print(ci_table.to_string(float_format="{:.2f}".format))

# %% [markdown]
r"""
## Completo contra LoRA: qué fracción entrenó y cuánto de la mejora compró

"Cuánto de la mejora compró" se define igual para cada métrica: `(LoRA − base) / (completo − base)`. 100%
significa que LoRA logró lo mismo que el fine-tuning completo; 0%, que no se movió del base. En las
pérdidas la mejora es una baja; en el comportamiento, una suba. Si el completo casi no mejoró sobre el
base en alguna métrica (menos de 0,05), ese cociente no significa nada y se marca como `nan`.
"""


# %%
def final_losses(model) -> dict[str, float]:
    return estimate_loss(model, FULL)


losses = {name: final_losses(model) for name, model in MODELS.items()}
ws = summary.xs("Words + Story", level="plantilla")
sw = summary.xs("solo Words", level="plantilla")
metrics = pd.DataFrame(
    {
        name: {
            "pérdida Instruct val": losses[name]["instruct_val"],
            "pérdida TinyStories liso val": losses[name]["plain_val"],
            "contenido: uso − piso (W+S)": ws.loc[name, "uso − piso"],
            "forma: arranca cuento (solo W)": sw.loc[name, "arranca cuento"],
            "forma: termina (W+S)": ws.loc[name, "termina"],
        }
        for name in MODELS
    }
)


def recovered(row: pd.Series) -> float:
    full_gain = row["completo"] - row["base"]
    if abs(full_gain) < 0.05:  # 0,05 nats o 5 puntos de fracción: por debajo, el cociente es ruido
        return float("nan")
    return (row["lora"] - row["base"]) / full_gain


metrics["LoRA / completo"] = metrics.apply(recovered, axis=1)

cost = pd.DataFrame({"completo": sft_cost, "lora": lora_cost})
cost.to_csv(OUTPUT_DIR / "costo_lora.csv")
metrics.to_csv(OUTPUT_DIR / "comparacion_lora.csv")
print(cost.to_string(float_format=lambda v: f"{v:,.4f}" if abs(v) < 1 else f"{v:,.1f}"))
print()
print(metrics.to_string(float_format="{:.3f}".format))

with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "smoke_test": SMOKE_TEST,
            "device": DEVICE,
            "base_checkpoint": str(BASE_PATH),
            "base_step": base_ckpt["step"],
            "n_train_records": len(train_records),
            "n_val_records": len(val_records),
            "story_marker": STORY_MARKER,
            "sft_full": asdict(FULL),
            "sft_lora": asdict(LORA),
            "n_samples": N_SAMPLES,
            "word_prompts": WORD_PROMPTS,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

# %% [markdown]
r"""
## Para el informe

Completar después de **la corrida completa**; lo que sale con `SMOKE_TEST` solo prueba que el código corre.

1. **El par antes/después con el mismo prompt: ¿qué cambió, la forma o el contenido?** Copien las tres
   salidas de `Words: dragon, happy, forest` (las dos plantillas). Léanlas antes de mirar la tabla, y
   después contrasten con `forma_contenido.csv`:
   - *Forma*: ¿el base, en `solo Words`, escribe `Story:` y un cuento, o sigue con más encabezados o con
     texto que no es un cuento? ¿El de SFT sí? ¿Terminan con `<|endoftext|>`?
   - *Contenido*: ¿el uso de palabras del SFT le gana a su propio piso de azar (`contenido_bootstrap.csv`)?
     ¿Y el base? Ojo: el base puede "usar" alguna palabra solo porque `forest` o `happy` son frecuentes en
     TinyStories; por eso se compara contra el piso y no contra cero.
   - Si se movió uno solo de los dos ejes, díganlo así, con el número. Miren el panel derecho de las
     curvas: ¿cuál de los dos se movió primero?
2. **¿Qué harían para conseguir la parte que no se movió?** Opciones concretas: más pasos o más registros
   (el contenido suele necesitar más que la forma: el formato se repite en *cada* ejemplo, cada palabra
   pedida aparece en pocos); enmascarar la pérdida de la instrucción para que todo el gradiente vaya al
   cuento; filtrar a registros con `Words:` para concentrar la señal; un modelo más profundo (la Etapa 2
   sugiere que el contexto largo es cosa de profundidad, y usar una palabra pedida 100 tokens antes es
   justamente contexto).
3. **LoRA: ¿qué fracción de los parámetros entrenaron, y cuánto de la mejora compró?** Usen
   `costo_lora.csv` (fracción entrenable, memoria pico al entrenar, segundos por paso) y la columna `LoRA / completo`
   de `comparacion_lora.csv`. Tengan en cuenta qué *no* toca LoRA acá: embeddings, MLP y `lm_head` quedan
   congelados, así que todo lo que aprenda tiene que pasar por *cómo se mira el contexto*. ¿Les alcanza
   eso para la forma? ¿Y para el contenido? ¿Y la memoria: el ahorro en gradientes + AdamW se nota en el
   pico, o a esta escala dominan las activaciones?
4. **Olvido.** ¿Cuánto subió la pérdida en TinyStories liso con cada método? LoRA, al no tocar los pesos
   base, ¿olvidó menos?
"""
