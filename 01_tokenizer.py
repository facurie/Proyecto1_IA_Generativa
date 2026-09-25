# %% [markdown]
r"""
# Etapa 1 · Tokenización — un BPE propio

Un tokenizador no es un default que se hereda: se entrena sobre *estos* datos, y el tamaño del
vocabulario (`V`) se paga contra el largo de las secuencias (`T`). Este notebook:

1. entrena un BPE a nivel byte sobre un subconjunto de TinyStories;
2. verifica que codificar → decodificar devuelva exactamente el texto original;
3. muestra los primeros merges que aprendió;
4. lo compara contra el tokenizador de GPT-2 (`tiktoken`, encoding `gpt2`) sobre cuentos que no vio;
5. barre el tamaño del vocabulario para ver el toma y daca `V ↔ T` con números;
6. mide cuánto de un cuento entra en cada `block_size` candidato — insumo directo de la Etapa 2.

**Produce:** `checkpoints/tokenizer.json`, que leen las Etapas 2 a 5.
"""

# %%
import environment  # noqa: F401  (primero siempre: caché de HF, stdout UTF-8, checkpoints/)

environment.summary()

# Entrenar un BPE es barato incluso en CPU (un par de minutos), así que acá la corrida completa es el default.
SMOKE_TEST = False
SEED = 1337

VOCAB_SIZE = 8_192
EOT = "<|endoftext|>"  # separa cuentos; el único token especial
N_TRAIN_STORIES = 10_000 if SMOKE_TEST else 100_000  # split train: de acá salen los merges
N_EVAL_STORIES = 500 if SMOKE_TEST else 5_000  # split validation: el tokenizador nunca lo ve
TOKENIZER_PATH = environment.CHECKPOINTS / "tokenizer.json"

# %%
from data import load_tinystories

train_texts = list(load_tinystories("train", limit=N_TRAIN_STORIES)["text"])
eval_texts = list(load_tinystories("validation", limit=N_EVAL_STORIES)["text"])

print(f"cuentos para entrenar el BPE : {len(train_texts):>7,}  (split train)")
print(f"cuentos para medir           : {len(eval_texts):>7,}  (split validation, held-out)")

# %% [markdown]
r"""
## Entrenar el BPE

A nivel byte, como GPT-2: el alfabeto inicial son los 256 bytes, así que cualquier string se puede
codificar y no hace falta un token `<unk>`. El pre-tokenizador `ByteLevel` corta en palabras y codifica el
espacio que las precede como `Ġ` (U+0120): el token de `" dog"` es el string `"Ġdog"`.
"""

# %%
import time

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def train_bpe(texts: list[str], vocab_size: int) -> Tokenizer:
    """BPE a nivel byte entrenado sobre `texts`, con `<|endoftext|>` como único token especial."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    batches = (texts[i : i + 1_000] for i in range(0, len(texts), 1_000))
    tokenizer.train_from_iterator(batches, trainer=trainer, length=len(texts))
    return tokenizer


t0 = time.perf_counter()
tokenizer = train_bpe(train_texts, VOCAB_SIZE)
print(f"vocabulario: {tokenizer.get_vocab_size():,} tokens · entrenado en {time.perf_counter() - t0:.1f}s")

# %% [markdown]
r"""
## Ida y vuelta

`decode(encode(s)) == s` tiene que valer para *todos* los cuentos held-out, no para un par elegidos a mano.
Si falla uno solo, hay un bug de configuración (típicamente, el decoder no coincide con el pre-tokenizador).
"""

# %%
failures = [s for s in eval_texts if tokenizer.decode(tokenizer.encode(s).ids) != s]
print(f"ida y vuelta exacta: {len(eval_texts) - len(failures):,}/{len(eval_texts):,} cuentos")
assert not failures, f"primer cuento que no vuelve igual: {failures[0][:200]!r}"

example = eval_texts[0]
encoding = tokenizer.encode(example)
print()
print(example[:300], "...")
print()
print(f"{len(encoding.ids)} tokens; los primeros 40:")
print(encoding.tokens[:40])

# %% [markdown]
r"""
## Qué aprendió primero

Los merges se aplican en el orden en que se aprendieron, y ese orden es el de frecuencia: los primeros
son los pares de símbolos más comunes de *este* corpus.
"""

# %%
import json

merges = json.loads(tokenizer.to_str())["model"]["merges"]
merges = [m.split(" ") if isinstance(m, str) else m for m in merges]  # "a b" o ["a", "b"] según la versión

print("  #  merge                   -> token nuevo")
for rank, (a, b) in enumerate(merges[:25], start=1):
    print(f"{rank:>3}  {a!r:>9} + {b!r:<10} -> {a + b!r}")

# %%
vocab = tokenizer.get_vocab()
whole_words = [t for t in vocab if t.startswith("Ġ") and t[1:].isalpha()]
longest = sorted(vocab, key=len, reverse=True)[:15]

print(f"tokens de palabra entera (Ġ + letras): {len(whole_words):,} de {len(vocab):,} "
      f"({len(whole_words) / len(vocab):.0%})")
print(f"los 15 tokens más largos: {longest}")

# %% [markdown]
r"""
## Contra GPT-2

Los mismos cuentos held-out, tres tokenizadores. Además de cuántos tokens produce cada uno, la última
columna muestra el otro lado de la balanza: lo que cuesta el vocabulario en parámetros del modelo. La tabla
de embeddings y la capa de salida son `V × n_embd` cada una (con `n_embd = 256`, el valor de la Etapa 2).
"""

# %%
import numpy as np
import pandas as pd
import tiktoken

N_EMBD_REFERENCE = 256

n_words = sum(len(s.split()) for s in eval_texts)
n_chars = sum(len(s) for s in eval_texts)

gpt2 = tiktoken.get_encoding("gpt2")
cl100k = tiktoken.get_encoding("cl100k_base")

tokens_per_story = {
    "BPE propio": (tokenizer.get_vocab_size(), [len(e.ids) for e in tokenizer.encode_batch(eval_texts)]),
    "GPT-2 (gpt2)": (gpt2.n_vocab, [len(ids) for ids in gpt2.encode_ordinary_batch(eval_texts)]),
    "GPT-3.5/4 (cl100k_base)": (cl100k.n_vocab, [len(ids) for ids in cl100k.encode_ordinary_batch(eval_texts)]),
}

ours_total = sum(tokens_per_story["BPE propio"][1])
comparison = pd.DataFrame(
    {
        name: {
            "V": vocab_size,
            "tokens/cuento (media)": np.mean(counts),
            "tokens/cuento (mediana)": np.median(counts),
            "tokens/palabra": sum(counts) / n_words,
            "caracteres/token": n_chars / sum(counts),
            "tokens vs. BPE propio": sum(counts) / ours_total,
            "params emb + salida (M)": 2 * vocab_size * N_EMBD_REFERENCE / 1e6,
        }
        for name, (vocab_size, counts) in tokens_per_story.items()
    }
).T
comparison["V"] = comparison["V"].astype(int)
print(f"{len(eval_texts):,} cuentos held-out · {n_words:,} palabras\n")
print(comparison.round(2).to_string())

# %% [markdown]
r"""
## El toma y daca `V ↔ T`, barrido

Un tokenizador por tamaño de vocabulario, todos sobre el mismo subconjunto (más chico que el de arriba,
así que el punto `V = 8.192` no coincide exacto con la tabla anterior). Para cada uno: cuántos tokens por
cuento, cuánto cuesta en parámetros, y qué fracción del vocabulario **no aparece nunca** en el held-out
— filas de embedding que casi no reciben gradiente (la trampa de la Etapa 3).
"""

# %%
SWEEP_VOCAB_SIZES = [512, 1_024, 2_048, 4_096, 8_192, 16_384]
SWEEP_TRAIN_STORIES = 5_000 if SMOKE_TEST else 20_000

sweep_rows = []
for vocab_size in SWEEP_VOCAB_SIZES:
    candidate = train_bpe(train_texts[:SWEEP_TRAIN_STORIES], vocab_size)
    ids = np.concatenate([e.ids for e in candidate.encode_batch(eval_texts)])
    usage = np.bincount(ids, minlength=candidate.get_vocab_size())
    sweep_rows.append(
        {
            "V": candidate.get_vocab_size(),
            "tokens/cuento": len(ids) / len(eval_texts),
            "params emb + salida (M)": 2 * candidate.get_vocab_size() * N_EMBD_REFERENCE / 1e6,
            "% vocab sin usar en held-out": 100 * (usage == 0).mean(),
        }
    )

sweep = pd.DataFrame(sweep_rows).set_index("V")
print(sweep.round(2).to_string())

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, column in zip(axes, sweep.columns):
    ax.plot(sweep.index, sweep[column], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("tamaño del vocabulario V")
    ax.set_title(column)
    ax.grid(alpha=0.3)
fig.suptitle(f"V ↔ T sobre TinyStories (n_embd = {N_EMBD_REFERENCE} para el costo en parámetros)")
fig.tight_layout()
fig.savefig(environment.CHECKPOINTS / "stage1_vocab_sweep.png", dpi=110)
plt.show()

# %% [markdown]
r"""
## Cuánto de un cuento entra en la ventana de contexto

Con el tokenizador elegido, el largo de cada cuento en tokens (+1 por el `<|endoftext|>` que lo separa del
siguiente). La Etapa 2 elige `block_size` entre 128 y 256: esto dice qué fracción de los cuentos el modelo
puede ver *enteros* dentro de una sola ventana.
"""

# %%
story_lengths = np.array([len(e.ids) + 1 for e in tokenizer.encode_batch(eval_texts)])

p50, p90, p99 = np.percentile(story_lengths, [50, 90, 99])
print(f"tokens por cuento · mediana {p50:.0f} · p90 {p90:.0f} · p99 {p99:.0f} · máx {story_lengths.max()}")
for block_size in (128, 192, 256):
    print(f"block_size {block_size}: {(story_lengths <= block_size).mean():6.1%} de los cuentos entran enteros")

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(story_lengths, bins=60, color="0.6")
for block_size, color in zip((128, 192, 256), ("C0", "C1", "C2")):
    ax.axvline(block_size, color=color, ls="--", label=f"block_size = {block_size}")
ax.set_xlabel("tokens por cuento (BPE propio)")
ax.set_ylabel("cantidad de cuentos")
ax.legend()
fig.tight_layout()
fig.savefig(environment.CHECKPOINTS / "stage1_story_lengths.png", dpi=110)
plt.show()

# %% [markdown]
r"""
## Guardar

Se guarda y se vuelve a cargar, para confirmar que el archivo reproduce exactamente la misma codificación.
"""

# %%
tokenizer.save(str(TOKENIZER_PATH))
reloaded = Tokenizer.from_file(str(TOKENIZER_PATH))
assert reloaded.encode(example).ids == encoding.ids
print(f"guardado: {TOKENIZER_PATH}  ({TOKENIZER_PATH.stat().st_size / 1e3:.0f} KB)")

# %% [markdown]
r"""
## Para el informe

- ¿Qué merges aparecieron primero, y qué dicen sobre cómo está armado este corpus? ¿Por qué *este* corpus
  produce *estos* merges y no, por ejemplo, un corpus de código?
- Con el vocabulario elegido, ¿cuántos tokens por cuento da en promedio, y cómo se compara con GPT-2 sobre
  los mismos cuentos? ¿Qué dice esa diferencia sobre el toma y daca `V ↔ T` en un corpus de vocabulario
  angosto?
- ¿Un vocabulario más chico es siempre mejor acá, o hay algo que se paga a cambio? (Mirar las tres curvas
  del barrido juntas.)

Lo que sigue: Etapa 2 (`02_pretraining.py`) — entrenar el Transformer sobre TinyStories con este
tokenizador.
"""
