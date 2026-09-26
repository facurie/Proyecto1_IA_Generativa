# %% [markdown]
r"""
# Etapa 3 · Análisis de embeddings aprendidos

Comparamos la tabla de embeddings de **la misma corrida** de la Etapa 2 antes y después
de entrenar. No reconstruimos el Transformer: leemos `token_embedding_table.weight`
directamente en CPU. Este análisis no entrena modelos ni descarga datasets.

**Entrada predeterminada:** `checkpoints/tokenizer.json`, `checkpoints/pretrain_step0.pt`
y `checkpoints/pretrain_wide.pt`. Para validar el funcionamiento con la corrida corta,
cambiá `SMOKE_TEST` a `True`: sólo los modelos pasan a `checkpoints/smoke/`.
No se activa automáticamente en CPU ni se sustituyen archivos faltantes.

**Salidas:** tablas CSV, gráficos PNG y `metadata.json` en `stage3/` dentro del directorio
de checkpoints elegido. Una nueva ejecución reemplaza esos resultados de análisis.
Ejecutá las celdas en orden desde la raíz del repositorio.

Los checkpoints actuales **no guardan la semilla ni una huella del tokenizer**.
Usá el tokenizer original y dos checkpoints de la misma corrida; las verificaciones
de tamaño y configuración no pueden demostrar esa procedencia. La semilla de abajo
controla el análisis, no certifica la semilla usada al entrenar.
"""

# %%
import environment  # configura rutas y stdout UTF-8; no carga datos ni entrena

import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from tokenizers import Tokenizer

SMOKE_TEST = environment.smoke_test(False)  # LAB_SMOKE_TEST=1 lo activa desde run_all.py
SEED = 1337
CKPT_DIR = environment.CHECKPOINTS / "smoke" if SMOKE_TEST else environment.CHECKPOINTS
TOKENIZER_PATH = environment.CHECKPOINTS / "tokenizer.json"
INITIAL_PATH = CKPT_DIR / "pretrain_step0.pt"
TRAINED_PATH = CKPT_DIR / "pretrain_wide.pt"  # editable si conservaste otro checkpoint
OUTPUT_DIR = CKPT_DIR / "stage3"

# Fijados antes de observar resultados. happy/sad comparten un eje semántico aunque
# sean antónimos: cercanía distribucional no equivale a sinonimia.
PAIRS = [("dog", "cat", "relacionado"), ("happy", "sad", "relacionado"),
         ("dog", "spoon", "esperado lejano")]
TOP_N = 10
N_RANDOM_PAIRS = 10_000
N_MAX_QUERIES = 512
N_PCA_WORDS = 500
MAX_BATCH_SIZE = 64
PERCENTILES = [1, 5, 50, 95, 99]
RUN_LABEL = "SMOKE · validación técnica" if SMOKE_TEST else "Corrida completa"
print(f"{RUN_LABEL} · CPU · semilla del análisis: {SEED}")

# %% [markdown]
r"""
## 1 · Cargar y verificar la comparación

Comprobamos configuraciones, dimensiones, valores finitos, IDs consecutivos y su
correspondencia con las filas; la referencia debe ser el paso cero y el modelo
entrenado debe estar en un paso posterior. También verificamos el historial disponible.

La Etapa 2 sobrescribe `pretrain_wide.pt` en cada evaluación: **último no significa mejor**.
Mostramos el paso cargado, su pérdida de validación y el mejor paso registrado. Si son
distintos, el historial no permite recuperar los pesos sobrescritos. Podés editar
`TRAINED_PATH` si guardaste el mejor checkpoint por separado. No inventamos una pérdida
para el paso cero si su archivo tiene el historial vacío.
"""

# %%
def require_files(paths):
    missing = [f"{name}: {path}" for name, path in paths.items() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Faltan artefactos:\n  " + "\n  ".join(missing)
            + "\nTraé pretrain_step0.pt y pretrain_wide.pt de la misma corrida de la Etapa 2, "
            "junto con su tokenizer.json original de la Etapa 1. "
            "Para una validación técnica, activá SMOKE_TEST explícitamente; no hay sustitución automática."
        )


def load_embedding_checkpoint(path):
    """Carga en CPU y conserva sólo la tabla y los metadatos necesarios."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    config = raw.get("model_config", {})
    required = {"vocab_size", "n_embd", "n_layer", "n_head", "block_size", "dropout"}
    if not required.issubset(config):
        raise ValueError(f"{path}: model_config incompleta; faltan {sorted(required - config.keys())}.")
    for key in required - {"dropout"}:
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{path}: dimensión inválida en model_config[{key!r}].")
    if config["n_embd"] % config["n_head"] or not 0 <= config["dropout"] < 1:
        raise ValueError(f"{path}: configuración de cabezas o dropout inválida.")
    weights = raw.get("state_dict", {}).get("token_embedding_table.weight")
    shape = (config["vocab_size"], config["n_embd"])
    if not isinstance(weights, torch.Tensor) or tuple(weights.shape) != shape:
        raise ValueError(f"{path}: token_embedding_table.weight debe tener forma {shape}.")
    if not weights.is_floating_point() or not torch.isfinite(weights).all():
        raise ValueError(f"{path}: embeddings no flotantes o con valores no finitos.")
    step = raw.get("step")
    if not isinstance(step, int) or step < 0:
        raise ValueError(f"{path}: paso inválido: {step!r}.")
    history = raw.get("history", [])
    previous_step = -1
    for row in history:
        logged_step, loss = row.get("step"), row.get("val_loss")
        if not isinstance(logged_step, int) or not previous_step < logged_step <= step:
            raise ValueError(f"{path}: pasos del historial inválidos o fuera del paso cargado.")
        if not isinstance(loss, (int, float)) or not np.isfinite(loss):
            raise ValueError(f"{path}: pérdida de validación no finita o ausente.")
        previous_step = logged_step
    return {
        "path": str(Path(path).resolve()), "model_config": config,
        "train_config": raw.get("train_config"), "step": step, "history": history,
        "weights": weights.detach().float().numpy().copy(),
    }


def validate_comparison(initial, trained, tokenizer):
    if initial["model_config"] != trained["model_config"]:
        raise ValueError("Las configuraciones de los modelos son incompatibles; usá la misma corrida wide.")
    if initial["train_config"] != trained["train_config"]:
        raise ValueError("Las configuraciones de entrenamiento difieren; revisá la procedencia de los archivos.")
    if initial["step"] != 0 or trained["step"] <= 0:
        raise ValueError("La referencia debe estar en el paso 0 y el modelo entrenado en un paso posterior.")
    vocab = tokenizer.get_vocab()
    size = initial["model_config"]["vocab_size"]
    if len(vocab) != size or sorted(vocab.values()) != list(range(size)):
        raise ValueError("Los IDs del tokenizer no corresponden uno a uno a las filas de embeddings.")
    if any(tokenizer.id_to_token(i) != token for token, i in vocab.items()):
        raise ValueError("Los IDs del tokenizer no permiten el ida y vuelta token → ID → token.")
    return vocab


def checkpoint_summary(checkpoint):
    history, step = checkpoint["history"], checkpoint["step"]
    best = min(history, key=lambda row: row["val_loss"]) if history else None
    current = next((row for row in history if row["step"] == step), None)
    is_best = current["val_loss"] == best["val_loss"] if current and best else None
    if best and not is_best:
        warnings.warn(
            f"{checkpoint['path']}: paso cargado {step}; mejor paso registrado {best['step']}. "
            "No se puede confirmar que los pesos cargados sean el mínimo del historial. "
            "El historial no permite recuperar pesos sobrescritos; editá TRAINED_PATH si conservaste otro archivo.",
            stacklevel=2,
        )
    if step > 0 and current is None:
        warnings.warn(f"No hay pérdida de validación registrada para el paso cargado {step}.", stacklevel=2)
    return {
        "path": checkpoint["path"], "step": step,
        "val_loss": current["val_loss"] if current else None,
        "best_step": best["step"] if best else None,
        "best_val_loss": best["val_loss"] if best else None,
        "is_best_recorded": is_best,
    }


require_files({"tokenizer": TOKENIZER_PATH, "inicial": INITIAL_PATH, "entrenado": TRAINED_PATH})
tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
initial = load_embedding_checkpoint(INITIAL_PATH)
trained = load_embedding_checkpoint(TRAINED_PATH)
vocab = validate_comparison(initial, trained, tokenizer)
checkpoint_info = {"inicial": checkpoint_summary(initial), "entrenado": checkpoint_summary(trained)}
print("Configuración del modelo:", initial["model_config"])
print("Configuración del entrenamiento:", trained["train_config"])
print(pd.DataFrame(checkpoint_info).T.to_string())
print("Procedencia a confirmar: misma corrida, misma inicialización y tokenizer original.")

# %% [markdown]
r"""
## 2 · Palabras enteras, pares y diez vecinos

Usamos exactamente `tok.startswith("Ġ") and tok[1:].isalpha()`. `Ġ` es U+0120, no un
espacio ASCII. Es un criterio operativo para seleccionar candidatos BPE, no una
garantía de que cada token sea una palabra completa en todos sus contextos.
El mapa palabra → posición se construye **exclusivamente** desde esos tokens,
sin mezclar `Ġday` con `day`, ni cambiar mayúsculas o minúsculas.

Normalizamos una vez cada matriz completa; sus filas filtradas conservan esa norma.
Los cosenos son productos vectorizados, y excluimos del ranking la propia fila de
consulta. Los pares elegidos antes de correr son **dog–cat**, **happy–sad** y **dog–spoon**.
Si falta alguno como token `Ġpalabra`, detenemos el análisis para revisar la configuración.
"""

# %%
def whole_word_index(vocab):
    candidates = sorted((tok, i) for tok, i in vocab.items() if tok.startswith("Ġ") and tok[1:].isalpha())
    if not candidates:
        raise ValueError("No hay candidatos de palabras enteras con el prefijo Ġ (U+0120).")
    tokens = [tok for tok, _ in candidates]
    rows = np.array([i for _, i in candidates], dtype=np.int64)
    positions = {tok[1:]: pos for pos, tok in enumerate(tokens)}
    return tokens, rows, positions


def query_positions(words, positions):
    missing = [word for word in words if word not in positions]
    if missing:
        raise ValueError(
            f"Palabra ausente como token de palabra entera: {missing}. "
            "Revisá PAIRS y el tokenizer original; no se usarán fragmentos sin Ġ."
        )
    return np.array([positions[word] for word in words], dtype=np.int64)


def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("No se puede calcular coseno: hay filas nulas o valores no finitos.")
    return matrix / norms


def nearest(unit, query_id, candidate_ids, topn):
    """Devuelve IDs originales y cosenos; la consulta siempre conserva su fila Ġ."""
    ids = candidate_ids[candidate_ids != query_id]
    similarities = np.clip(unit[ids] @ unit[query_id], -1.0, 1.0)
    order = np.argsort(-similarities, kind="stable")[:topn]
    return ids[order], similarities[order]


def pair_cosines(unit, pairs):
    return np.clip(np.einsum("ij,ij->i", unit[pairs[:, 0]], unit[pairs[:, 1]]), -1.0, 1.0)


def neighbor_table(units, query_words, query_ids, candidate_ids, tokenizer, topn, pool_name):
    records = []
    for state, unit in units.items():
        for word, query_id in zip(query_words, query_ids):
            ids, scores = nearest(unit, query_id, candidate_ids, topn)
            for rank, (token_id, score) in enumerate(zip(ids, scores), start=1):
                token = tokenizer.id_to_token(int(token_id))
                records.append({
                    "estado": state, "candidatos": pool_name, "consulta": word,
                    "consulta_id": int(query_id), "rango": rank, "token": token,
                    "token_id": int(token_id), "coseno": float(score),
                    "pasa_filtro": token.startswith("Ġ") and token[1:].isalpha(),
                })
    return pd.DataFrame(records)


whole_tokens, whole_rows, word_positions = whole_word_index(vocab)
words = [token[1:] for token in whole_tokens]
query_words = list(dict.fromkeys(word for a, b, _ in PAIRS for word in (a, b)))
query_pos = query_positions(query_words, word_positions)
query_ids = whole_rows[query_pos]
if len(whole_rows) < max(N_PCA_WORDS, N_MAX_QUERIES, TOP_N + 1):
    raise ValueError("No hay suficientes palabras para las muestras configuradas; revisá los tamaños de muestra.")
units = {"inicial": normalize_rows(initial["weights"]), "entrenado": normalize_rows(trained["weights"])}
word_units = {state: unit[whole_rows] for state, unit in units.items()}
chosen_positions = np.array([[word_positions[a], word_positions[b]] for a, b, _ in PAIRS])
chosen_ids = whole_rows[chosen_positions]
pair_table = pd.DataFrame(PAIRS, columns=["palabra_a", "palabra_b", "expectativa"])
pair_table["id_a"], pair_table["id_b"] = chosen_ids.T
for state, unit in units.items():
    pair_table[f"coseno_{state}"] = pair_cosines(unit, chosen_ids)
pair_table["delta"] = pair_table["coseno_entrenado"] - pair_table["coseno_inicial"]
whole_neighbors = neighbor_table(units, query_words, query_ids, whole_rows, tokenizer, TOP_N, "palabras enteras")
print(f"{len(vocab):,} tokens; {len(words):,} candidatos de palabras enteras.")
print(pair_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print(whole_neighbors.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# %% [markdown]
r"""
## 3 · Dos pisos de ruido distintos

**Un par fijado de antemano:** para vectores independientes e isotrópicos en dimensión
`n_embd`, `1/√n_embd` orienta la escala del coseno. No es un umbral de significancia.
Medimos además 10.000 pares aleatorios distintos de palabras enteras, sin autorrelaciones
ni ninguno de los pares elegidos (tampoco invertidos). Usamos **exactamente los mismos
IDs** antes y después de entrenar, para comparar sus cambios.

**Un vecino encontrado buscando:** elegimos 512 consultas reproducibles del modelo
inicial y buscamos el máximo coseno de cada una contra **todos** los candidatos
filtrados, excluyendo la propia consulta. La mediana y el percentil 95 de esos máximos
muestran cómo buscar entre muchos candidatos eleva el mejor resultado esperable por azar.

Las dos referencias responden preguntas distintas. Los percentiles son descriptivos,
no p-valores ni pruebas concluyentes de significado: los pares pueden compartir palabras.
Si todo el conjunto de controles aumenta su similitud, ese aumento común no es evidencia
específica de que los pares elegidos hayan adquirido una relación semántica.
"""

# %%
def sample_random_pairs(n_words, count, excluded, seed):
    """Muestreo uniforme sin reemplazo de pares no ordenados, por rechazo."""
    excluded = {tuple(sorted(pair)) for pair in excluded}
    available = n_words * (n_words - 1) // 2 - len(excluded)
    if count < 1 or count > available:
        raise ValueError(f"Se pidieron {count} controles, pero sólo hay {available} pares disponibles.")
    rng = np.random.default_rng(seed)
    selected = {}
    while len(selected) < count:
        batch = np.sort(rng.integers(0, n_words, size=(max(256, count - len(selected)), 2)), axis=1)
        for a, b in batch:
            pair = (int(a), int(b))
            if a != b and pair not in excluded:
                selected.setdefault(pair, None)
            if len(selected) == count:
                break
    return np.array(list(selected), dtype=np.int64)


def maximum_cosines(unit, positions, batch_size=64):
    """Máximos contra todo el pool; bloques para no materializar V × V."""
    maxima = []
    for start in range(0, len(positions), batch_size):
        batch = positions[start:start + batch_size]
        similarities = unit[batch] @ unit.T
        similarities[np.arange(len(batch)), batch] = -np.inf
        maxima.extend(np.clip(similarities.max(axis=1), -1.0, 1.0))
    return np.array(maxima)


excluded_pairs = {tuple(pair) for pair in chosen_positions}
control_positions = sample_random_pairs(len(words), N_RANDOM_PAIRS, excluded_pairs, SEED)
control_ids = whole_rows[control_positions]
controls = pd.DataFrame({
    "palabra_a": np.array(words)[control_positions[:, 0]],
    "palabra_b": np.array(words)[control_positions[:, 1]],
    "id_a": control_ids[:, 0], "id_b": control_ids[:, 1],
})
for state, unit in units.items():
    controls[f"coseno_{state}"] = pair_cosines(unit, control_ids)
controls["delta"] = controls["coseno_entrenado"] - controls["coseno_inicial"]

percentile_records = []
for measure in ["coseno_inicial", "coseno_entrenado", "delta"]:
    values = controls[measure].to_numpy()
    percentile_records.append({
        "medida": measure, "media": float(values.mean()), "desvio": float(values.std()),
        **{f"p{p:02d}": float(np.percentile(values, p)) for p in PERCENTILES},
    })
    pair_table[f"percentil_{measure}"] = [100 * float(np.mean(values <= value)) for value in pair_table[measure]]
control_percentiles = pd.DataFrame(percentile_records)

max_query_pos = np.random.default_rng(SEED + 1).choice(len(words), N_MAX_QUERIES, replace=False)
maxima = maximum_cosines(word_units["inicial"], max_query_pos, MAX_BATCH_SIZE)
maxima_table = pd.DataFrame({"palabra": np.array(words)[max_query_pos], "token_id": whole_rows[max_query_pos], "max_coseno_inicial": maxima})
noise_scale = 1 / np.sqrt(initial["model_config"]["n_embd"])
noise_info = {"escala_1_sqrt_d": float(noise_scale), "mediana_maximos": float(np.median(maxima)),
              "p95_maximos": float(np.percentile(maxima, 95)), "n_candidatos_por_consulta": len(words) - 1}
print("Referencias de ruido:", noise_info)
print(control_percentiles.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("Percentil empírico = porcentaje de controles con valor <= al del par (no es un p-valor).")
print(pair_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
pair_colors = ["#0072B2", "#D55E00", "#009E73"]
fig_noise, noise_axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
for ax, measure, title in zip(noise_axes.flat, ["coseno_inicial", "coseno_entrenado", "delta"],
                              ["Pares fijados · inicial", "Pares fijados · entrenado", "Cambio entrenado − inicial"]):
    values = controls[measure].to_numpy()
    ax.hist(values, bins=60, color="#a8b6c5", label="10.000 controles" if N_RANDOM_PAIRS == 10_000 else f"{N_RANDOM_PAIRS} controles")
    for index, row in pair_table.iterrows():
        ax.axvline(row[measure], color=pair_colors[index % len(pair_colors)], linewidth=1.8,
                   label=f"{row.palabra_a}–{row.palabra_b}: {row[measure]:.3f}")
    for sign in [-1, 1] if measure == "coseno_inicial" else []:
        ax.axvline(sign * noise_scale, color="black", ls="--", alpha=0.7,
                   label="±1/√d (escala)" if sign == 1 else None)
    ax.set(title=title, xlabel="Δ coseno" if measure == "delta" else "Coseno", ylabel="Cantidad de pares")
    ax.legend(fontsize=8)
ax = noise_axes[1, 1]
ax.hist(maxima, bins=35, color="#a8b6c5")
for key, style, label in [("mediana_maximos", "--", "Mediana"), ("p95_maximos", ":", "P95")]:
    ax.axvline(noise_info[key], color="black", ls=style, label=f"{label}: {noise_info[key]:.3f}")
ax.set(title=f"Búsqueda inicial · {N_MAX_QUERIES} consultas, {len(words) - 1:,} candidatos",
       xlabel="Máximo coseno (sin la propia consulta)", ylabel="Cantidad de consultas")
ax.legend(fontsize=9)
fig_noise.suptitle(f"{RUN_LABEL} · controles y referencias descriptivas de ruido")
fig_noise.savefig(OUTPUT_DIR / "controles_ruido.png", dpi=160)
plt.show()

# %% [markdown]
r"""
## 4 · Una proyección comparable

Elegimos 500 palabras sin reemplazo e incluimos siempre las consultas. Ajustamos **una
única PCA** sobre la concatenación de sus embeddings normalizados iniciales y entrenados,
y transformamos ambos estados con esos mismos ejes. Dos PCA independientes podrían
rotar los paneles de forma distinta y dificultar la comparación.

Los dos paneles tienen iguales límites, escala, colores y etiquetas. Sólo etiquetamos
las consultas para conservar la legibilidad. La varianza explicada indica qué fracción
retienen estos dos ejes: cercanía en 2D no prueba cercanía en todas las dimensiones.
La proyección es exploratoria; las tablas de cosenos son la referencia cuantitativa.
"""

# %%
if len(query_pos) > N_PCA_WORDS:
    raise ValueError("N_PCA_WORDS debe alcanzar para incluir todas las consultas.")
remaining = np.setdiff1d(np.arange(len(words)), query_pos)
extra = np.random.default_rng(SEED + 2).choice(remaining, N_PCA_WORDS - len(query_pos), replace=False)
pca_positions = np.sort(np.concatenate([query_pos, extra]))
pca_ids = whole_rows[pca_positions]
joint = np.concatenate([units[state][pca_ids] for state in ["inicial", "entrenado"]])
pca = PCA(n_components=2, svd_solver="full")
pca.fit(joint)
projected = {state: pca.transform(units[state][pca_ids]) for state in ["inicial", "entrenado"]}
variance = pca.explained_variance_ratio_
print(f"Varianza explicada: PC1={variance[0]:.2%}, PC2={variance[1]:.2%}, total={variance.sum():.2%}.")

pca_table = pd.concat([
    pd.DataFrame({"estado": state, "palabra": np.array(words)[pca_positions], "token_id": pca_ids,
                  "pc1": coords[:, 0], "pc2": coords[:, 1], "es_consulta": np.isin(pca_positions, query_pos)})
    for state, coords in projected.items()
], ignore_index=True)
all_coords = np.concatenate(list(projected.values()))
minimum, maximum = all_coords.min(axis=0), all_coords.max(axis=0)
margin = np.maximum(maximum - minimum, 1e-6) * 0.15
query_palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
query_colors = {word: query_palette[i % len(query_palette)] for i, word in enumerate(query_words)}
fig_pca, pca_axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True, constrained_layout=True)
for ax, (state, coords) in zip(pca_axes, projected.items()):
    ax.scatter(coords[:, 0], coords[:, 1], s=12, color="#9da9b5", alpha=0.4)
    for i, (word, position) in enumerate(zip(query_words, query_pos)):
        x, y = coords[np.flatnonzero(pca_positions == position)[0]]
        ax.scatter(x, y, s=55, color=query_colors[word], edgecolor="white", linewidth=0.5)
        ax.annotate(word, (x, y), xytext=(6, 8 if i % 2 == 0 else -13), textcoords="offset points",
                    color=query_colors[word], fontsize=10)
    ax.set(title=f"{state.capitalize()} · paso {checkpoint_info[state]['step']}",
           xlabel=f"PC1 ({variance[0]:.2%})", ylabel=f"PC2 ({variance[1]:.2%})",
           xlim=(minimum[0] - margin[0], maximum[0] + margin[0]),
           ylim=(minimum[1] - margin[1], maximum[1] + margin[1]))
    ax.set_aspect("equal", adjustable="box")
fig_pca.suptitle(f"{RUN_LABEL} · PCA conjunta · {N_PCA_WORDS} palabras · varianza total {variance.sum():.2%}")
fig_pca.savefig(OUTPUT_DIR / "pca_comparada.png", dpi=160)
plt.show()

# %% [markdown]
r"""
## 5 · Qué cambia al incluir fragmentos

Repetimos las búsquedas con **la misma fila `Ġpalabra` como consulta**, ahora contra todo
el vocabulario. Mostramos los tokens originales (incluido `Ġ`) y la proporción de los
diez vecinos que pasa el filtro. Al ampliar el conjunto de candidatos también cambia
la selección del máximo; no atribuyamos toda diferencia a semántica.

`Ġday` y `day` son filas distintas. Medimos su coseno y mostramos ambos IDs, sin
atribuirles frecuencias que no medimos: el checkpoint no contiene conteos de uso de
tokens. La presencia de un fragmento entre los vecinos, por sí sola, no demuestra que
haya quedado sin entrenar.
"""

# %%
all_neighbors = neighbor_table(units, query_words, query_ids, np.arange(len(vocab)), tokenizer, TOP_N, "todo el vocabulario")
filter_neighbors = pd.concat([whole_neighbors, all_neighbors], ignore_index=True)
filter_summary = filter_neighbors.groupby(["estado", "consulta", "candidatos"], sort=False).agg(
    n_vecinos=("token_id", "size"), proporcion_palabras_enteras=("pasa_filtro", "mean"),
).reset_index()
print(all_neighbors.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(filter_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

day_records = []
for token in ["Ġday", "day"]:
    token_id = vocab.get(token)
    day_records.append({"token": token, "token_id": token_id, "presente": token_id is not None,
                        "pasa_filtro": token.startswith("Ġ") and token[1:].isalpha()})
day_table = pd.DataFrame(day_records)
if all(record["presente"] for record in day_records):
    day_ids = np.array([[vocab["Ġday"], vocab["day"]]])
    for state, unit in units.items():
        day_table[f"coseno_entre_filas_{state}"] = float(pair_cosines(unit, day_ids)[0])
else:
    print("Este tokenizer no contiene ambas variantes de day; no se sustituirán por otros tokens.")
print(day_table.to_string(index=False))

# %% [markdown]
r"""
## 6 · Guardar resultados y trazabilidad

Guardamos los IDs de cada par de control, consulta del piso de máximos y palabra de la
PCA. El archivo de metadatos registra las rutas, pasos, configuraciones, semillas y
versiones de paquetes. La huella del tokenizer que guardamos **ahora** sirve para
identificar esta ejecución; no demuestra qué tokenizer produjo los checkpoints viejos.

Los CSV y PNG de una corrida `SMOKE_TEST` se rotulan como **validación técnica**. Las
conclusiones sobre el aprendizaje de la corrida completa quedan pendientes de correr
con sus artefactos y examinar los resultados.
"""

# %%
tables = {
    "pares": pair_table, "vecinos_palabras": whole_neighbors, "controles_pares": controls,
    "controles_percentiles": control_percentiles, "maximos_iniciales": maxima_table,
    "piso_ruido": pd.DataFrame([noise_info]), "pca_coordenadas": pca_table,
    "vecinos_filtrado": filter_neighbors, "resumen_filtrado": filter_summary,
    "ejemplo_day": day_table,
}
for name, table in tables.items():
    numeric = table.select_dtypes(include="number")
    # Sólo el ID de una variante de day ausente puede quedar vacío; las mediciones no.
    if name == "ejemplo_day":
        numeric = numeric.drop(columns=["token_id"], errors="ignore")
    numeric = numeric.to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError(f"La tabla {name} contiene resultados no finitos.")
    exported = table.copy()
    exported.insert(0, "tipo_corrida", RUN_LABEL)
    exported.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8")

metadata = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_label": RUN_LABEL,
    "smoke_test": SMOKE_TEST, "device": "cpu", "seed": SEED,
    "paths": {"tokenizer": str(TOKENIZER_PATH.resolve()), "initial": str(INITIAL_PATH.resolve()),
              "trained": str(TRAINED_PATH.resolve()), "output": str(OUTPUT_DIR.resolve())},
    "tokenizer_sha256_at_analysis": hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest(),
    "provenance_limit": "Los checkpoints no registran semilla ni huella del tokenizer; verificar misma corrida y tokenizer original.",
    "checkpoints": checkpoint_info, "model_config": initial["model_config"],
    "train_config": trained["train_config"], "trained_history": trained["history"],
    "analysis": {
        "pairs": PAIRS, "top_n": TOP_N, "n_whole_words": len(words),
        "whole_word_filter": 'tok.startswith("Ġ") and tok[1:].isalpha()',
        "n_random_pairs": N_RANDOM_PAIRS, "random_pairs_seed": SEED,
        "random_pairs_csv": "controles_pares.csv", "percentiles": PERCENTILES,
        "percentile_definition": "100 * mean(control <= observed); descriptivo, no p-valor",
        "n_max_queries": N_MAX_QUERIES, "max_queries_seed": SEED + 1,
        "max_query_ids": whole_rows[max_query_pos].tolist(), "max_batch_size": MAX_BATCH_SIZE,
        "n_pca_words": N_PCA_WORDS, "pca_sample_seed": SEED + 2,
        "pca_sample_ids": pca_ids.tolist(), "pca_sample_words": np.array(words)[pca_positions].tolist(),
        "pca_fit": "ambos estados concatenados, embeddings normalizados, svd_solver=full",
        "pca_explained_variance_ratio": variance.tolist(), "query_colors": query_colors,
        "noise": noise_info,
    },
    "versions": {"python": platform.python_version(), **{package: version(package) for package in
                 ["numpy", "pandas", "torch", "tokenizers", "scikit-learn", "matplotlib"]}},
    "artifacts": [f"{name}.csv" for name in tables] + ["controles_ruido.png", "pca_comparada.png", "metadata.json"],
}
(OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
print(f"{RUN_LABEL}: {len(tables)} tablas, 2 gráficos y metadatos guardados en {OUTPUT_DIR.resolve()}")

# %% [markdown]
r"""
## 7 · Las cuatro preguntas para el informe

Completá estas respuestas después de ejecutar **la corrida completa**. Registrá el paso
analizado y si coincide con el mejor disponible. Un resultado débil, bien delimitado,
también responde la consigna; no se requiere que estos pares confirmen la expectativa.

1. **Dos pares cerca y uno lejos: ¿le pegaron y la diferencia le gana al ruido?**
   Usá `pares.csv`, `vecinos_palabras.csv`, `controles_percentiles.csv`, `piso_ruido.csv`
   y `controles_ruido.png`. Compará tanto el coseno inicial/entrenado como el cambio
   contra los mismos controles. Para juzgar vecinos encontrados buscando, usá la
   distribución de máximos iniciales, no sólo `1/√n_embd`. El P95 orienta la lectura;
   superarlo no prueba semántica. ¿dog–cat y happy–sad se desplazan más que los controles?
   ¿dog–spoon contradice la predicción? Evitá elegir nuevos pares después de mirar y
   presentarlos como si se hubieran fijado antes.

2. **¿Qué cambió a la vista entre las proyecciones 2D?**
   Usá `pca_comparada.png` y `pca_coordenadas.csv`, con iguales ejes y la varianza
   explicada indicada. Describí cambios visibles y contrastalos con cosenos en la
   dimensión original. La superposición o separación en dos ejes puede esconder
   cambios en dimensiones descartadas.

3. **¿Filtrar palabras enteras cambia lo que ven?**
   Contrastá `vecinos_filtrado.csv` y `resumen_filtrado.csv`: la consulta sigue siendo
   la misma fila `Ġpalabra`. Usá `ejemplo_day.csv` para mostrar que `Ġday` y `day`
   son IDs diferentes. Discutí la composición de vecinos y el tamaño del conjunto de
   candidatos; no infieras frecuencias de uso ni filas sin entrenar sin medirlas.

4. **Si el espacio entrenado se parece al inicial, ¿qué cambiarían y cuánto costaría?**
   Primero distinguí falta de evidencia en estas mediciones de ausencia de aprendizaje:
   estos embeddings son estáticos, mientras que el modelo también aprende en las capas
   posteriores. Considerá más pasos (más cómputo y posible sobreajuste), más datos
   diversos y pertinentes (tokenización, almacenamiento y entrenamiento adicionales),
   o distinta dimensión (cambia capacidad, memoria, costo por paso y escala `1/√d`).
   Cambiá una variable por vez, guardá una nueva referencia inicial de esa corrida y
   recalculá sus controles. Medí el costo junto con la pérdida de validación antes de
   atribuir una mejora al cambio. Para afirmar robustez, repetir con otras semillas
   también cuesta corridas adicionales.

**Conclusiones de la corrida completa:** pendientes de ejecutar y redactar con evidencia
propia. Los resultados rotulados SMOKE sólo validan que este análisis funciona.
"""
