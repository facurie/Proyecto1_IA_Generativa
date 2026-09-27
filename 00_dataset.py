# %% [markdown]
r"""
# Etapa 0 · El dataset como instrumento

Este notebook es el punto de partida: preparación del entorno y una primera mirada a los datos. Todavía no hay modelado.

TinyStories (Eldan & Li, 2023, arXiv:2305.07759) son cuentos cortos restringidos a un vocabulario que un
chico de 3-4 años entiende. Esa restricción no es una limitación que haya que sortear -- es lo que convierte
al dataset en un instrumento de laboratorio: aísla la variable "datos" del ruido de la escala, de modo que un
modelo de <10M de parámetros puede producir inglés coherente en lugar de balbuceo.
"""

# %%
import environment  # noqa: F401  (efectos colaterales: caché de HF, bundle de certificados, ...)

environment.summary()

SMOKE_TEST = True  # True: una porción mínima, corre en segundos. False: una muestra de inspección más grande.
SEED = 1337

# %% [markdown]
r"""
## Cargar los datasets

Dos datasets, ambos traídos del HuggingFace Hub, nunca incluidos dentro de este repositorio:

- `roneneldan/TinyStories` -- el corpus de preentrenamiento (Etapa 2). Un dataset Parquet estándar, se corta en
  porciones de forma barata.
- `roneneldan/TinyStories-Instruct` -- el corpus de instruction-tuning (Etapa 4). Cada registro
  antepone un SUBCONJUNTO aleatorio, en ORDEN aleatorio, de hasta cuatro tipos de instrucción (`Features:`,
  `Words:`, `Summary:`, `Random sentence:`) antes del cuento -- confirmado muestreando varios
  miles de registros, no asumido. **Trampa:** este es un dataset legacy con script de carga, almacenado
  con una LÍNEA por fila, no un ejemplo por fila, y `datasets` materializa el split completo de ~21,7M de
  líneas antes de poder cortar nada -- esperá un costo único de ~20-30s en la primera corrida (después queda
  cacheado, no es que se colgó). `data.load_instruct_records` reagrupa las líneas en registros completos por vos; mirá el
  docstring de ese módulo para entender por qué el mirror de la comunidad (`skeskinen/TinyStories-Instruct-hf`) es peor
  acá, no mejor -- directamente rompe en una ruta de clonado profunda de Windows.
"""

# %%
from data import load_instruct_records, load_tinystories

INSPECT_N = 1_000 if SMOKE_TEST else 20_000

stories = load_tinystories(limit=INSPECT_N)
print(stories)
print()
print("--- un cuento crudo ---")
print(stories[0]["text"])

# %%
instruct_records = load_instruct_records(limit=200 if SMOKE_TEST else 2_000)
print(f"registros instruct parseados: {len(instruct_records)}")
print()
print("--- un registro instruct crudo ---")
print(instruct_records[0])

# %% [markdown]
r"""
## Distribución de longitudes

Un proxy por cantidad de palabras alcanza acá (la Etapa 1 te va a dar un conteo exacto de tokens una vez que tengas un
tokenizador). Esto importa para la Etapa 2: te dice qué parte de un cuento típico cubre realmente el
`block_size` (largo del contexto).
"""

# %%
import matplotlib.pyplot as plt

lengths = [len(s.split()) for s in stories["text"]]
print(f"cuentos inspeccionados   : {len(lengths)}")
print(f"palabras/cuento (media)  : {sum(lengths) / len(lengths):.1f}")
print(f"palabras/cuento min / max: {min(lengths)} / {max(lengths)}")

plt.figure(figsize=(6, 3))
plt.hist(lengths, bins=40)
plt.xlabel("palabras por cuento")
plt.ylabel("cantidad")
plt.title(f"Distribución de longitudes de TinyStories (n={len(lengths)})")
plt.tight_layout()
plt.savefig("checkpoints/stage0_length_hist.png", dpi=100)
plt.show()

# %% [markdown]
r"""
## Un tamaño aproximado del vocabulario

Separando por espacios y pasando a minúsculas -- no es un tokenizador de verdad (eso es la Etapa 1), apenas lo suficiente
para sostener la afirmación del "vocabulario acotado" con un número en lugar de una intuición.
"""

# %%
from collections import Counter

word_counts = Counter(w.lower().strip(".,!?;:\"'") for s in stories["text"] for w in s.split())
print(f"palabras únicas (aprox.) : {len(word_counts)}")
print(f"las 10 más comunes       : {word_counts.most_common(10)}")

# %% [markdown]
r"""
## Preguntas para el informe

**¿Qué les da experimentalmente "el vocabulario de un chico de 3–4 años" que un corpus general de texto web no les da?**
En los 1.000 cuentos que miramos encontramos unas 4.961 palabras distintas y un promedio de 183,8
palabras por cuento. Como el vocabulario es chico y los cuentos tienen un estilo parecido, el modelo ve
muchas veces las mismas palabras y formas de escribir. Con pocos parámetros y la misma cantidad de
datos, eso le facilita aprender a armar cuentos que se entiendan. Nos sirve para estudiar qué puede
aprender con datos simples y repetidos. Igual, que escriba bien estos cuentos no significa que entienda
cualquier tema.

**Si entrenaran la misma arquitectura sobre un pedazo de texto crudo de internet con la misma cantidad de tokens, ¿qué esperarían que cambie, y por qué?**
Si usamos texto web, el modelo vería más temas, palabras poco comunes, nombres, enlaces y hasta restos
de HTML. Entonces tendría menos ejemplos repetidos de cada forma de escribir. Además, si dejamos el
mismo tokenizador, algunas palabras se separarían en más tokens y ocuparían más espacio del contexto.
Esperaríamos que le cueste más escribir cuentos cortos y coherentes como los de TinyStories, aunque esto
habría que probarlo. Para compararlos bien, usaríamos el mismo tokenizador y los mismos textos de prueba;
no alcanza con mirar la pérdida de cada modelo en su propio corpus porque son textos diferentes.
"""
