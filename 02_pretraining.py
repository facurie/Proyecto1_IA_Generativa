# %% [markdown]
r"""
# Etapa 2 · Preentrenamiento — misma arquitectura, mejores datos

El Transformer decoder-only de siempre (`Head` → `MultiHeadAttention` → `FeedForward` → `Block`), con la
tabla de embeddings apuntando al vocabulario BPE de la Etapa 1 en lugar de a caracteres, entrenado sobre
TinyStories. El ciclo `forward → loss → backward → update` no cambia; lo nuevo son los datos.

Encima de la corrida base, **la ablación ancho contra profundo**: dos modelos con una cantidad de
parámetros parecida, uno con `n_embd` grande y pocas capas, otro con `n_embd` más chico y más capas,
entrenados con exactamente los mismos datos, pasos, semilla y optimizador.

**Lee:** `checkpoints/tokenizer.json` (Etapa 1).
**Produce:** `checkpoints/pretrain_step0.pt`, `checkpoints/pretrain_wide.pt`, `checkpoints/pretrain_deep.pt`.

**Cómo correrlo.** En CPU, `SMOKE_TEST` se activa solo: valida el pipeline en minutos (~200 pasos) y
guarda en `checkpoints/smoke/`, así nunca pisa un checkpoint de verdad. En la T4 de Colab hace la corrida
completa. Cada entrenamiento guarda un checkpoint en cada evaluación: si la sesión se corta, volver a
correr la celda **retoma desde ahí**, y si el checkpoint ya está completo, lo carga sin reentrenar.
"""

# %%
import environment  # noqa: F401  (primero siempre: caché de HF, stdout UTF-8, checkpoints/)

environment.summary()

import hashlib
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field

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
CKPT_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER_PATH = environment.CHECKPOINTS / "tokenizer.json"

# Mixed precision en la GPU: la T4 tiene tensor cores en float16 y no en float32.
USE_AMP = DEVICE == "cuda"

torch.manual_seed(SEED)
print(f"dispositivo: {DEVICE} · SMOKE_TEST: {SMOKE_TEST} · AMP: {USE_AMP} · checkpoints en: {CKPT_DIR}")

# %% [markdown]
r"""
## Datos

Cada cuento se tokeniza con el BPE de la Etapa 1 y se le agrega `<|endoftext|>` al final; todo se
concatena en un único arreglo largo de ids. Un batch son ventanas al azar de `block_size + 1` tokens de
ese arreglo: `x` son los primeros `block_size` y `y` el mismo tramo corrido en uno.

La pérdida held-out sale del split `validation` de TinyStories, que el modelo nunca ve.
"""

# %%
if not TOKENIZER_PATH.is_file():
    raise FileNotFoundError(
        f"No está {TOKENIZER_PATH}. Corré primero la Etapa 1 (01_tokenizer) en esta misma sesión, "
        "o traé el archivo de Drive."
    )

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
VOCAB_SIZE = tokenizer.get_vocab_size()
EOT_ID = tokenizer.token_to_id("<|endoftext|>")
# Huella del tokenizer: va en cada checkpoint. Volver a correr la Etapa 1 pisa tokenizer.json, y un modelo
# entrenado con otro tokenizer carga sin error pero lee ids que ya no significan lo mismo.
TOKENIZER_SHA1 = hashlib.sha1(TOKENIZER_PATH.read_bytes()).hexdigest()

N_TRAIN_STORIES = 5_000 if SMOKE_TEST else 400_000
N_VAL_STORIES = 500 if SMOKE_TEST else 10_000


def encode_corpus(texts: list[str], chunk: int = 10_000) -> np.ndarray:
    """
    Todos los cuentos en un solo arreglo de ids, cada uno seguido de `<|endoftext|>`.

    `uint16` alcanza para V ≤ 65.535 y ocupa la cuarta parte que `int64`: 400k cuentos son ~160 MB en
    lugar de ~640 MB. Se pasa a `int64` recién al armar cada batch.
    """
    assert VOCAB_SIZE <= np.iinfo(np.uint16).max
    parts = []
    for i in range(0, len(texts), chunk):
        encodings = tokenizer.encode_batch(texts[i : i + chunk])
        parts.append(np.concatenate([np.array(e.ids + [EOT_ID], dtype=np.uint16) for e in encodings]))
    return np.concatenate(parts)


t0 = time.perf_counter()
train_ids = encode_corpus(list(load_tinystories("train", limit=N_TRAIN_STORIES)["text"]))
val_ids = encode_corpus(list(load_tinystories("validation", limit=N_VAL_STORIES)["text"]))
print(f"vocabulario: {VOCAB_SIZE:,} · <|endoftext|> = id {EOT_ID}")
print(f"train: {N_TRAIN_STORIES:>7,} cuentos → {len(train_ids) / 1e6:6.2f}M tokens")
print(f"val  : {N_VAL_STORIES:>7,} cuentos → {len(val_ids) / 1e6:6.2f}M tokens")
print(f"tokenizado en {time.perf_counter() - t0:.0f}s")

# %% [markdown]
r"""
## El modelo

El mismo GPT de la clase. Tres decisiones que conviene tener a la vista:

- **Sin weight tying:** la tabla de embeddings y el `lm_head` son matrices separadas, las dos de
  `V × n_embd`. Con `V = 8.192` son el grueso del modelo, y eso es lo que hace tramposa la ablación.
- **Inicialización `N(0, 0.02)`** como en GPT-2: logits iniciales chicos, así la pérdida del paso 0 arranca
  en ≈ `ln(V)`, el costo de adivinar uniforme sobre el vocabulario.
- **`generate` recibe un `torch.Generator` propio** en lugar de usar el RNG global: se puede fijar la
  semilla de cada muestra sin tocar el azar del entrenamiento (el dropout) y sin caer en la trampa de
  sembrar el RNG global adentro del loop.
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
        # No persistente: la máscara se reconstruye sola, no tiene por qué ocupar lugar en el checkpoint.
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


def count_params(model: GPTLanguageModel) -> dict[str, int]:
    """Parámetros por grupo, para ver dónde está el grueso del modelo."""
    groups = {
        "embeddings (token + posición)": [model.token_embedding_table, model.position_embedding_table],
        "bloques Transformer": [model.blocks],
        "salida (ln_f + lm_head)": [model.ln_f, model.lm_head],
    }
    counts = {name: sum(p.numel() for m in modules for p in m.parameters()) for name, modules in groups.items()}
    counts["total"] = sum(p.numel() for p in model.parameters())
    return counts


# %% [markdown]
r"""
## El par de la ablación, contado antes de entrenar

Partir el ancho al medio y duplicar la profundidad **no** deja el tamaño fijo: los embeddings y el
`lm_head` escalan con `V × n_embd`, así que bajar `n_embd` achica el grueso del modelo y sumar capas solo
agranda el resto. Si los totales difieren mucho, el experimento pasa a ser "grande contra chico".

Decisiones de diseño, para el registro:

- **`block_size = 256`:** según la Etapa 1, con 256 el 90% de los cuentos held-out entra entero en una
  ventana, contra el 60% con 192. Para probar consistencia a lo largo de un cuento, conviene ver el cuento.
- **`head_size = 32` en los dos modelos** (8 cabezas de 32 en el ancho, 6 de 32 en el profundo): así lo
  único que cambia entre ellos es el ancho del residual y la cantidad de capas, no el tamaño de cada cabeza.
"""

# %%
BLOCK_SIZE = 256

CONFIGS = {
    "wide": GPTConfig(VOCAB_SIZE, BLOCK_SIZE, n_embd=256, n_head=8, n_layer=2),
    "deep": GPTConfig(VOCAB_SIZE, BLOCK_SIZE, n_embd=192, n_head=6, n_layer=5),
}

param_table = pd.DataFrame({name: count_params(GPTLanguageModel(cfg)) for name, cfg in CONFIGS.items()})
param_table["deep / wide"] = param_table["deep"] / param_table["wide"]
print(param_table.to_string(formatters={"wide": "{:,}".format, "deep": "{:,}".format, "deep / wide": "{:.1%}".format}))

totals = param_table.loc["total", ["wide", "deep"]]
print(f"\ndiferencia de tamaño: {abs(totals['wide'] - totals['deep']) / totals.max():.1%}")
print(f"pérdida esperada en el paso 0 ≈ ln(V) = {math.log(VOCAB_SIZE):.2f}")

# %% [markdown]
r"""
## Entrenamiento

AdamW con warmup lineal y decaimiento coseno, clipping de gradiente, y mixed precision en la GPU. Las dos
corridas de la ablación comparten todo esto; solo cambia `GPTConfig`.

- **Evaluación sobre batches fijos:** cada evaluación usa los mismos batches (misma semilla), así las
  curvas se mueven por lo que cambió el modelo y no por qué ventanas tocaron.
- **Muestras durante el entrenamiento:** en los pasos de `sample_steps`, una continuación de
  `"Once upon a time"` con la misma semilla, para leer cuándo aparece la gramática y cuándo la coherencia.
- **Checkpoints:** en cada evaluación se guarda el modelo con su configuración, el historial, las muestras
  y el estado del optimizador. Al retomar, la continuación no es idéntica bit a bit a una corrida sin
  cortes (el RNG del dropout no se restaura), pero sí equivalente.
"""


# %%
@dataclass
class TrainConfig:
    max_steps: int
    batch_size: int
    eval_interval: int
    eval_iters: int
    sample_steps: tuple[int, ...]
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0


if SMOKE_TEST:
    TRAIN = TrainConfig(max_steps=200, batch_size=16, eval_interval=50, eval_iters=10, sample_steps=(0, 100, 200))
else:
    TRAIN = TrainConfig(
        max_steps=5_000,
        batch_size=64,
        eval_interval=250,
        eval_iters=50,
        sample_steps=(0, 250, 500, 1_000, 2_000, 3_500, 5_000),
    )

SAMPLE_PROMPT = "Once upon a time"

tokens_per_run = TRAIN.max_steps * TRAIN.batch_size * BLOCK_SIZE
print(TRAIN)
print(f"tokens vistos por corrida: {tokens_per_run / 1e6:.1f}M ≈ {tokens_per_run / len(train_ids):.2f} épocas")


# %%
def autocast():
    return torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP)


def get_batch(data: np.ndarray, batch_size: int, block_size: int, generator: torch.Generator):
    """Ventanas al azar de block_size + 1 tokens: x son los primeros block_size, y el mismo tramo corrido en uno."""
    starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator).tolist()
    windows = torch.from_numpy(np.stack([data[i : i + block_size + 1] for i in starts]).astype(np.int64))
    x = windows[:, :-1].contiguous().to(DEVICE)
    y = windows[:, 1:].contiguous().to(DEVICE)
    return x, y


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Warmup lineal hasta `lr` y después decaimiento coseno hasta `min_lr`."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model: GPTLanguageModel, cfg: TrainConfig) -> dict[str, float]:
    """Pérdida media en train y val, siempre sobre los mismos `eval_iters` batches."""
    was_training = model.training
    model.eval()
    losses = {}
    for split, data in (("train", train_ids), ("val", val_ids)):
        generator = torch.Generator().manual_seed(SEED)
        batch_losses = []
        for _ in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.batch_size, model.cfg.block_size, generator)
            with autocast():
                _, loss = model(x, y)
            batch_losses.append(loss.item())
        losses[split] = float(np.mean(batch_losses))
    model.train(was_training)
    return losses


def encode_prompt(prompt: str) -> list[int]:
    """
    Antepone `<|endoftext|>`: en el corpus, cada cuento viene después de uno, así que eso pone al modelo
    en "principio de cuento", que es la situación que aprendió.
    """
    return [EOT_ID] + tokenizer.encode(prompt).ids


def generate_text(model, prompt, *, max_new_tokens=200, greedy=False, temperature=1.0, seed=SEED) -> str:
    """Una continuación de `prompt` (sin el prompt), muestreada con su propio generador sembrado con `seed`."""
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    idx = torch.tensor([encode_prompt(prompt)], device=DEVICE)
    with autocast():
        out = model.generate(
            idx, max_new_tokens, temperature=temperature, greedy=greedy, generator=generator, stop_token=EOT_ID
        )
    model.train(was_training)
    return tokenizer.decode(out[0, idx.shape[1] :].tolist())


# %%
def save_checkpoint(path, model, *, step, train_cfg, history, samples, optimizer=None, scaler=None, batch_rng=None):
    """
    Guarda `state_dict` + la configuración que define la forma del modelo (lo mínimo para reconstruirlo) y,
    si se pasa el optimizador, todo lo necesario para retomar el entrenamiento. Se escribe a un temporal y
    después se renombra: un corte a mitad de la escritura no deja un checkpoint roto.
    """
    checkpoint = {
        "model_config": asdict(model.cfg),
        "state_dict": model.state_dict(),
        "step": step,
        "train_config": asdict(train_cfg),
        "history": history,
        "samples": samples,
        "tokenizer_sha1": TOKENIZER_SHA1,
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
        checkpoint["scaler"] = scaler.state_dict()
        checkpoint["batch_rng"] = batch_rng.get_state()
    tmp = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)


def load_model(path, device=DEVICE):
    """Reconstruye un modelo a partir de un checkpoint. Devuelve (modelo en modo eval, checkpoint crudo)."""
    checkpoint = torch.load(path, map_location=device)
    model = GPTLanguageModel(GPTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


@dataclass
class Run:
    name: str
    model: GPTLanguageModel
    history: list[dict] = field(default_factory=list)
    samples: dict[int, str] = field(default_factory=dict)


def train(name: str, cfg: GPTConfig, train_cfg: TrainConfig, *, step0_path=None) -> Run:
    """
    Entrena un modelo desde cero, o retoma desde `pretrain_<name>.pt` si existe.

    Con `step0_path`, antes de dar un solo paso guarda el modelo sin entrenar (misma semilla): es la
    referencia de la Etapa 3.
    """
    assert all(s % train_cfg.eval_interval == 0 or s == train_cfg.max_steps for s in train_cfg.sample_steps), (
        "cada paso de sample_steps tiene que coincidir con una evaluación"
    )
    path = CKPT_DIR / f"pretrain_{name}.pt"

    torch.manual_seed(SEED)  # misma inicialización para cualquier corrida con la misma config
    model = GPTLanguageModel(cfg).to(DEVICE)
    if step0_path is not None and not step0_path.exists():
        save_checkpoint(step0_path, model, step=0, train_cfg=train_cfg, history=[], samples={})
        print(f"[{name}] guardado el modelo sin entrenar en {step0_path.name}")

    # Weight decay solo sobre matrices; ni bias ni LayerNorm.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": train_cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=train_cfg.lr,
    )
    scaler = torch.amp.GradScaler(DEVICE, enabled=USE_AMP)
    batch_rng = torch.Generator().manual_seed(SEED)  # el orden de los batches, separado del RNG global
    run = Run(name, model)
    start = 0

    if path.exists():
        # A CPU: el estado de `batch_rng` tiene que quedar ahí; modelo y optimizador se mueven solos al cargarse.
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint["model_config"] != asdict(cfg) or checkpoint["train_config"] != asdict(train_cfg):
            raise ValueError(
                f"{path} es de otra configuración. Borralo o renombralo para entrenar de cero con esta."
            )
        if checkpoint.get("tokenizer_sha1", TOKENIZER_SHA1) != TOKENIZER_SHA1:
            raise ValueError(
                f"{path} se entrenó con otro tokenizer.json (¿se volvió a correr la Etapa 1?). "
                "Borralo para reentrenar con el tokenizer actual, o traé el tokenizer original."
            )
        model.load_state_dict(checkpoint["state_dict"])
        run.history, run.samples, start = checkpoint["history"], checkpoint["samples"], checkpoint["step"]
        if start >= train_cfg.max_steps:
            print(f"[{name}] ya estaba entrenado ({start:,} pasos): se carga {path.name} sin reentrenar.")
            return run
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        batch_rng.set_state(checkpoint["batch_rng"])
        print(f"[{name}] retomando desde el paso {start:,}.")

    tokens_per_step = train_cfg.batch_size * cfg.block_size
    elapsed_before = run.history[-1]["elapsed_s"] if run.history else 0.0
    t0 = time.perf_counter()
    model.train()

    for step in range(start, train_cfg.max_steps + 1):
        is_eval_step = step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps
        already_logged = bool(run.history) and run.history[-1]["step"] == step  # el paso en que se retomó
        if is_eval_step and not already_logged:
            losses = estimate_loss(model, train_cfg)
            elapsed = elapsed_before + time.perf_counter() - t0
            run.history.append(
                {
                    "step": step,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "val_ppl": math.exp(losses["val"]),
                    "lr": lr_at(step, train_cfg),
                    "tokens_seen": step * tokens_per_step,
                    "elapsed_s": elapsed,
                }
            )
            print(
                f"[{name}] paso {step:>5,} | train {losses['train']:.3f} | val {losses['val']:.3f} "
                f"| ppl {math.exp(losses['val']):8.1f} | {elapsed / 60:5.1f} min"
            )
            if step in train_cfg.sample_steps:
                run.samples[step] = generate_text(model, SAMPLE_PROMPT, max_new_tokens=120)
            save_checkpoint(
                path, model, step=step, train_cfg=train_cfg, history=run.history, samples=run.samples,
                optimizer=optimizer, scaler=scaler, batch_rng=batch_rng,
            )
        if step == train_cfg.max_steps:
            break

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, train_cfg)
        x, y = get_batch(train_ids, train_cfg.batch_size, cfg.block_size, batch_rng)
        with autocast():
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

    return run


runs: dict[str, Run] = {}

# %% [markdown]
r"""
### Modelo ancho (`n_embd = 256`, `n_layer = 2`)

Es también el que leen las Etapas 3 y 4, así que acá se guarda además el modelo del paso 0.
"""

# %%
runs["wide"] = train("wide", CONFIGS["wide"], TRAIN, step0_path=CKPT_DIR / "pretrain_step0.pt")

# %% [markdown]
r"""
### Modelo profundo (`n_embd = 192`, `n_layer = 5`)
"""

# %%
runs["deep"] = train("deep", CONFIGS["deep"], TRAIN)

# %% [markdown]
r"""
## Curvas de pérdida y perplejidad

La línea punteada es `ln(V)`: la pérdida de adivinar uniforme sobre el vocabulario. Si en unos cientos de
pasos la curva no baja claramente de ahí, el problema está en los datos o la tokenización, no en los
hiperparámetros.
"""

# %%
COLORS = {"wide": "C0", "deep": "C1"}

fig, (ax_loss, ax_ppl) = plt.subplots(1, 2, figsize=(12, 4))
for name, run in runs.items():
    steps = [h["step"] for h in run.history]
    ax_loss.plot(steps, [h["train_loss"] for h in run.history], color=COLORS[name], ls="--", label=f"{name} · train")
    ax_loss.plot(steps, [h["val_loss"] for h in run.history], color=COLORS[name], label=f"{name} · val")
    ax_ppl.plot(steps, [h["val_ppl"] for h in run.history], color=COLORS[name], label=name)

ax_loss.axhline(math.log(VOCAB_SIZE), color="0.5", ls=":", lw=1, label="ln(V)")
ax_loss.set(xlabel="paso", ylabel="entropía cruzada", title="Pérdida")
ax_ppl.set(xlabel="paso", ylabel="perplejidad (val)", title="Perplejidad held-out", yscale="log")
for ax in (ax_loss, ax_ppl):
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
fig.savefig(CKPT_DIR / "stage2_curves.png", dpi=110)
plt.show()

# %% [markdown]
r"""
## Muestras a lo largo del entrenamiento

La misma semilla en todos los pasos: lo que cambia entre una muestra y la siguiente es el modelo, no el
azar. ¿En qué rango de pérdida el texto deja de ser ruido y empieza a tener forma de oración? ¿Cuándo se
sostiene un personaje a lo largo de varias oraciones?
"""

# %%
for name, run in runs.items():
    val_loss_at = {h["step"]: h["val_loss"] for h in run.history}
    print("=" * 100)
    print(f"{name.upper()} · {SAMPLE_PROMPT!r}")
    for step, text in sorted(run.samples.items()):
        print(f"\n--- paso {step:,} · val loss {val_loss_at[step]:.2f} ---")
        print(SAMPLE_PROMPT + text)
    print()

# %% [markdown]
r"""
## Prompts de prueba

Dos familias, para contrastar con lo que dice el paper (ancho ↔ conocimiento, profundidad ↔ contexto).
**Este set es un punto de partida: adáptenlo y defiendan sus propios prompts en el informe.**

- **Recuerdo:** completar un dato común. Se mide con decodificación *greedy* (la predicción más probable
  del modelo, no una muestra) y además con la probabilidad total que el modelo le da a las respuestas
  aceptables como próximo token, que es una medida continua y menos ruidosa que acertar o no.
- **Consistencia:** el prompt introduce un nombre y un objeto; se generan varias continuaciones (semilla
  `SEED + 100·i + j`: distinta por prompt y por muestra, idéntica entre modelos) y se mide qué fracción de
  las palabras clave vuelve a aparecer. Esto mide *mención*, no *uso correcto*: hay que leer las muestras.
"""

# %%
RECALL_PROBES = [
    ("The sky is", {"blue"}),
    ("The grass is", {"green"}),
    ("At night, the moon and the", {"stars"}),
    ("The fish swam in the", {"water", "pond", "sea", "lake", "river", "ocean"}),
    ("Bees make", {"honey"}),
    ("When it rains, you need an", {"umbrella"}),
    ("Snow is cold and", {"white", "wet"}),
    ("The fire was very", {"hot"}),
    ("Birds can", {"fly", "sing"}),
    ("Apples are red and bananas are", {"yellow"}),
]

CONSISTENCY_PROBES = [
    ("Once upon a time, there was a girl named Mia. Mia had a red ball that she loved very much. One day,", ["Mia", "ball"]),
    ("Tom had a little dog named Max. Every morning, Tom and Max went to the park together. One day,", ["Tom", "Max"]),
    ("Anna found a shiny key in the garden. She wanted to know what it could open.", ["Anna", "key"]),
    ("Ben and his mom went to the beach. Ben wanted to build a big sand castle.", ["Ben", "castle"]),
    ("There was a little bird named Pip. Pip lost its blue feather in the wind.", ["Pip", "feather"]),
]
N_CONSISTENCY_SAMPLES = 2 if SMOKE_TEST else 5


def single_token_ids(words: set[str]) -> dict[str, int]:
    """Id de `Ġword` para cada respuesta que es un solo token de palabra entera; las demás se descartan."""
    ids = {w: tokenizer.token_to_id("Ġ" + w) for w in words}
    return {w: i for w, i in ids.items() if i is not None}


for prompt, answers in RECALL_PROBES:
    missing = answers - single_token_ids(answers).keys()
    if missing:
        print(f"aviso: {sorted(missing)} no es un solo token; no suma a P(respuesta) en {prompt!r}")


@torch.no_grad()
def run_recall_probes(model: GPTLanguageModel) -> pd.DataFrame:
    model.eval()
    rows = []
    for prompt, answers in RECALL_PROBES:
        idx = torch.tensor([encode_prompt(prompt)], device=DEVICE)
        logits, _ = model(idx)
        probs = F.softmax(logits[0, -1].float(), dim=-1)
        continuation = generate_text(model, prompt, max_new_tokens=6, greedy=True)
        first_word = re.findall(r"[A-Za-z]+", continuation)[:1]
        rows.append(
            {
                "prompt": prompt,
                "greedy": continuation.strip(),
                "acierto": bool(first_word) and first_word[0].lower() in answers,
                "P(respuesta)": sum(probs[i].item() for i in single_token_ids(answers).values()),
            }
        )
    return pd.DataFrame(rows)


def run_consistency_probes(model: GPTLanguageModel) -> pd.DataFrame:
    rows = []
    for i, (prompt, keywords) in enumerate(CONSISTENCY_PROBES):
        for j in range(N_CONSISTENCY_SAMPLES):
            text = generate_text(model, prompt, max_new_tokens=150, seed=SEED + 100 * i + j)
            reused = [k for k in keywords if re.search(rf"\b{re.escape(k)}s?\b", text, flags=re.IGNORECASE)]
            rows.append({"probe": i, "sample": j, "reuso": len(reused) / len(keywords), "texto": text})
    return pd.DataFrame(rows)


recall = {name: run_recall_probes(run.model) for name, run in runs.items()}
consistency = {name: run_consistency_probes(run.model) for name, run in runs.items()}

# %%
with pd.option_context("display.max_colwidth", 60, "display.width", 200):
    for name in runs:
        print(f"=== {name.upper()} · recuerdo ===")
        print(recall[name].to_string(index=False, float_format="{:.3f}".format))
        print()

# %%
for i, (prompt, keywords) in enumerate(CONSISTENCY_PROBES):
    print("=" * 100)
    print(f"[{i}] {prompt}   (palabras clave: {', '.join(keywords)})")
    for name in runs:
        first = consistency[name].query("probe == @i and sample == 0").iloc[0]
        print(f"\n--- {name} · reuso {first['reuso']:.0%} ---")
        print(first["texto"].strip())
    print()

# %% [markdown]
r"""
## La tabla de la ablación
"""

# %%
ablation = pd.DataFrame(
    {
        name: {
            "n_embd": run.model.cfg.n_embd,
            "n_layer": run.model.cfg.n_layer,
            "parámetros (M)": count_params(run.model)["total"] / 1e6,
            "train loss": run.history[-1]["train_loss"],
            "val loss": run.history[-1]["val_loss"],
            "val ppl": run.history[-1]["val_ppl"],
            "recuerdo: aciertos greedy": recall[name]["acierto"].mean(),
            "recuerdo: P(respuesta) media": recall[name]["P(respuesta)"].mean(),
            "consistencia: reuso medio": consistency[name]["reuso"].mean(),
            "minutos de entrenamiento": run.history[-1]["elapsed_s"] / 60,
        }
        for name, run in runs.items()
    }
)
ablation.to_csv(CKPT_DIR / "stage2_ablation.csv")
print(ablation.to_string(float_format="{:.3f}".format))

# %% [markdown]
r"""
## ¿Cuántos datos vieron, contra el óptimo de Chinchilla?

Hoffmann et al. (2022) estiman que, para un presupuesto de cómputo fijo, lo óptimo es entrenar con
≈ 20 tokens por parámetro. Kaplan et al. (2020) cuentan los parámetros *sin* embeddings (en sus modelos la
salida comparte la matriz de embeddings); acá eso equivale a sacar embeddings y `lm_head`, y con este
vocabulario la diferencia entre las dos cuentas es grande, así que se muestran ambas.
"""

# %%
for name, run in runs.items():
    counts = count_params(run.model)
    n_total = counts["total"]
    n_core = counts["bloques Transformer"]
    seen = run.history[-1]["tokens_seen"]
    print(f"{name}: {seen / 1e6:.1f}M tokens vistos ({seen / len(train_ids):.2f} épocas)")
    for label, n in (("N total", n_total), ("N sin embeddings ni lm_head", n_core)):
        print(f"    {label:<28} {n / 1e6:5.2f}M → 20·N = {20 * n / 1e6:4.0f}M tokens · vistos / óptimo = {seen / (20 * n):.2f}")

# %% [markdown]
r"""
## Para el informe

- ¿En qué rango de pérdida el texto deja de ser ruido y empieza a tener estructura de oración? (Cruzar las
  muestras con las curvas.)
- ¿El par ancho/profundo reproduce lo que encontró el paper (ancho ↔ conocimiento, profundidad ↔
  contexto)? Si no, ¿cuál es la mejor explicación: escala, cuánto entrenaron, cómo diseñaron los prompts?
- ¿Qué tuvieron que recortar para que la ablación entrara en la máquina que tenían, y qué habrían medido
  con más?
- Contra Chinchilla, ¿dónde quedaron parados y qué creen que les costó?

Lo que sigue: Etapa 3 (`03_embeddings.py`) — la tabla de embeddings de `pretrain_wide.pt` contra la de
`pretrain_step0.pt`.
"""
