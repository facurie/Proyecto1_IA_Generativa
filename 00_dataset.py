# %% [markdown]
r"""
# Etapa 0 · El dataset como instrumento

Este notebook es el punto de partida: preparación del entorno y una primera mirada a los datos. Todavía no hay modelado.

TinyStories (Eldan & Li, 2023, arXiv:2305.07759) son cuentos cortos restringidos a un vocabulario que un
chico de 3-4 años entiende. Esa restricción no es una limitación que haya que sortear -- es lo que convierte
al dataset en un instrumento de laboratorio: aísla la variable "datos" del ruido de la escala, de modo que un
modelo de <10M de parámetros puede producir inglés coherente en lugar de balbuceo.
"""

# %% [markdown]
r"""
## Cómo leer esta entrega

Son seis notebooks, uno por etapa, **todos ejecutados** en una notebook con RTX 5070 Laptop de 8 GB (semilla 1337;
las Etapas 1 a 5 a escala completa). El informe está en markdown dentro de los mismos notebooks: cada etapa cierra
con las respuestas a sus preguntas de la consigna, sus decisiones de diseño y sus dudas. Las mismas conclusiones,
junto con el registro de decisiones y el uso de IA, están reunidas en `informe_conclusiones.pdf`.

| Notebook | Qué hace | Dónde está el informe |
|---|---|---|
| `00_dataset` | Mira TinyStories y TinyStories-Instruct | Al final: respuestas de la Etapa 0, **registro de decisiones** del proyecto y **uso de IA** |
| `01_tokenizer` | BPE propio a nivel byte (`V = 8.192`) contra GPT-2; barrido de `V`; cuánto de un cuento entra en la ventana | Una lectura debajo de cada resultado; respuestas, decisiones, dudas y anexo al final |
| `02_pretraining` | El GPT de la clase sobre TinyStories; ablación ancho (256 × 2 capas) contra profundo (192 × 5); Chinchilla | "Informe y conclusiones de la Etapa 2", al final |
| `03_embeddings` | Embeddings del paso 0 contra el paso 5.000: pares, piso de ruido con controles, PCA | Secciones 7 y 8 |
| `04_sft` | SFT completo y LoRA a mano sobre TinyStories-Instruct; forma contra contenido | "Resultados de la corrida completa", al final |
| `05_judge` | Juez local (Ollama + `qwen3:4b`) calibrado con anclas; juez contra perplejidad | "Resultados de la corrida completa" y **cierre del proyecto**, al final |

**El hilo que atraviesa todo**, y que defendemos en el cierre del `05`: en nuestros modelos la *forma* aparece antes
que el *contenido*. Pasa en el preentrenamiento (la gramática llega antes que la consistencia), en el SFT (el formato
se instala en unos cientos de pasos; usar las palabras pedidas, a medias) y en la evaluación (la perplejidad mide
cuánto se parece un texto al corpus, no si es un buen cuento).

**Para correrlos en Colab** (GPU T4): la primera celda de cada notebook clona el repositorio del grupo e instala
`requirements.txt`. El repositorio ya trae el `tokenizer.json` y los checkpoints entrenados, así que las Etapas 2 y 4
los cargan en lugar de reentrenar (borrando `checkpoints/pretrain_*.pt` o `checkpoints/sft_*.pt` se entrena de cero).
No hace falta volver a correr el `01`: regenera el tokenizer, y si su huella no coincidiera con la que guardan los
checkpoints, las Etapas 2, 4 y 5 se frenan a propósito con un error explicativo en lugar de cargar un modelo que lee
ids con otro significado. La Etapa 5 instala Ollama y baja `qwen3:4b` (~2,5 GB).
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
## Lectura de lo que medimos

Esta inspección corrió con el valor por defecto del notebook (`SMOKE_TEST = True`: 1.000 cuentos del split `train` y
200 registros de Instruct). Alcanza para ver el formato y el orden de magnitud; las medidas finas, en tokens, están
en la Etapa 1, sobre 5.000 cuentos held-out y 100.000 de entrenamiento.

- **Largo:** 183,8 palabras por cuento en promedio, con el grueso entre 100 y 250 y una cola larga a la derecha
  (mínimo 61, máximo 837). En tokens de nuestro BPE (Etapa 1, sobre 5.000 cuentos held-out del split
  `validation`): 196 por cuento en promedio, mediana 181, y con `block_size = 256` entra entero el 90 %.
  **Un detalle que se nos había pasado:** los cuentos de `train` son más largos que los de `validation` (183,8
  palabras contra 158,5 en nuestras muestras; 221 tokens por cuento contra 201 en los datos de la Etapa 2, contando
  el `<|endoftext|>`), así que ese 90 % es optimista para los datos con los que efectivamente entrenamos.
- **Vocabulario:** unas 4.961 palabras distintas en ~184.000 palabras de texto, y las diez más comunes (*the, and,
  to, a, was, she, he, they, it, her*) son, solas, el 28,5 % de todo lo que se lee. Son palabras de función de un
  relato en pasado y en tercera persona: *was* aparece más que cualquier sustantivo.
- **Instruct:** el registro de ejemplo trae `Features: Dialogue`, `Words: quit, oak, gloomy` y un `Summary:` antes
  de `Story:`, y el cuento usa de verdad las tres palabras pedidas (*oak*, tres veces). Esa asociación instrucción →
  cuento es la que la Etapa 4 intenta enseñar.
"""

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

# %% [markdown]
r"""
## Ampliación: el mismo argumento, con los números de todas las etapas

Las respuestas de arriba, apoyadas en lo que medimos en las Etapas 1 a 5, y la pregunta con la que la consigna
abre esta etapa: por qué un vocabulario chico a propósito convierte al dataset en un instrumento de laboratorio.

### Por qué un vocabulario chico a propósito es un instrumento de laboratorio

**Porque achica el problema hasta que un modelo de nuestro tamaño deja de estar desbordado, y entonces lo que aprende
—o no aprende— se puede atribuir a los datos y no a que "le faltó escala".**

Un experimento sirve cuando se mueve una variable y el resto queda quieto. Con texto general, un modelo de 6M de
parámetros fallaría por todo a la vez: demasiadas palabras, temas, registros y datos del mundo para su tamaño. Ese
fracaso no diría nada, porque no habría forma de separar el efecto de los datos del de la escala. TinyStories reduce
lo que hay que aprender (léxico, sintaxis y un mundo de pocos personajes y objetos) a algo que entra en el modelo, y
recién ahí se puede preguntar qué aprende primero, qué necesita más capas o qué cambia con el SFT. Es la lógica de los
organismos modelo en biología: la mosca de la fruta no es "un animal cualquiera", es uno lo bastante simple como para
experimentar y lo bastante completo como para que el fenómeno aparezca.

Nuestros números lo muestran: con 5,85M de parámetros y 6 minutos de GPU, el modelo ancho pasa de adivinar al azar
(pérdida 9,06 ≈ ln 8.192) a sus primeras oraciones bien formadas en unos 500 pasos (pérdida 3,3), y termina en
perplejidad 10,1 (Etapa 2).

### Qué nos da el vocabulario chico, con números

1. **Repetición.** Pocas palabras se repiten muchísimo: diez palabras son el 28,5 % del texto y, con nuestro BPE, los
   1.000 tokens más usados cubren el 90 % (Etapa 1). Casi todas las filas de la tabla de embeddings reciben miles de
   ejemplos, y por eso en la Etapa 3 los vecinos de *dog* o de *sad* terminan teniendo sentido.
2. **Un tokenizador que es casi un diccionario.** Con 8.192 entradas, solo el 0,9 % de las palabras se parte en más
   de un token (Etapa 1): los tokens son palabras, se pueden analizar como palabras (Etapa 3), y un cuento entero entra
   en la ventana de 256.
3. **Un solo género.** Narración en pasado, en tercera persona, con diálogo y fórmulas fijas: el 67 % de los cuentos
   held-out empieza con *Once upon a time* (Etapa 1). Con una estructura global tan regular, la coherencia pasa a ser
   algo que un modelo chico puede llegar a aprender, y algo que se puede medir (Etapas 2 y 5).
4. **Un mundo chico.** Pocos personajes y objetos, causas simples. Eso permite preguntar por consistencia (¿el
   personaje sigue siendo el mismo? ¿usa las palabras que le pedimos?) sin que la respuesta dependa de conocimiento
   enciclopédico que un modelo de 6M no puede guardar.
5. **Experimentos baratos.** El pipeline entero corre en poco más de media hora en la GPU de una notebook. Eso es lo
   que nos dejó hacer la ablación, el SFT, LoRA y el juez, en lugar de elegir uno.

### Lo que no le creemos del todo

"El vocabulario de un chico de 3–4 años" es una intención del generador, no una
restricción dura. El paper armó los cuentos pidiéndole a GPT-3.5/4 que combinara palabras de una lista de unas 1.500
básicas, pero en solo 1.000 cuentos aparecen unas 5.000 formas distintas (con flexiones y nombres propios), y en
nuestro vocabulario BPE entraron *accomplishment*, *determination* y *compassionate* (Etapa 1). El corpus trae además
impurezas de su origen sintético: el 6 % de los cuentos tiene texto mal codificado (`â€™` en lugar de `’`), que el
modelo aprende como cualquier otro patrón. Y la misma regularidad que lo hace útil baja la vara: "coherente" acá quiere
decir coherente para un cuento infantil de fórmula, y parte del éxito de un modelo chico es haber aprendido la fórmula.

### La predicción sobre texto web, con lo que medimos

**Texto con la pinta de internet pero sin hilo: fluidez local, poca coherencia, y un modelo que gasta casi todo su
tamaño en cubrir palabras que ve pocas veces.** No lo corrimos; es una predicción, apoyada en lo que sí medimos:

- **El mismo presupuesto de tokens compraría menos texto.** Un BPE de 8.192 entradas entrenado sobre la web tiene que
  repartirse entre muchas más palabras, nombres, números, código y otros idiomas, así que parte más palabras: más
  tokens por palabra, menos palabras en los mismos 82M tokens y menos documento en cada ventana de 256. El tokenizador
  de GPT-2, entrenado sobre texto web, ya muestra ese reparto: entre sus primeros merges están los de la prosa
  informativa (`ion`, `ic`, *of*), que en el nuestro aparecen cientos de merges después (Etapa 1, anexo B).
- **Cada palabra se vería muchas menos veces.** En la web la cola larga es mucho más pesada: muchas filas de la tabla
  de embeddings quedarían casi sin entrenar (la trampa de *SolidGoldMagikarp* que discutimos en la Etapa 1), y la
  tabla de embeddings más la capa de salida ya son el 73 % de nuestro modelo ancho.
- **Mezcla de registros.** Noticias, listas, foros, código: el modelo repartiría su capacidad en imitar formatos
  distintos. Esperaríamos algo parecido a lo que vimos en clase con el Transformer de caracteres sobre Shakespeare: la
  pinta visual del texto, sin sentido.
- **La pérdida no se podría comparar directo.** Como decimos arriba, habría que medir a los dos modelos sobre los
  mismos textos de prueba; y si cambia el tokenizador, la pérdida por token cambia de escala y hay que pasarla a
  bits por carácter (Etapa 1, respuesta 3).
- **Y el laboratorio dejaría de funcionar.** Si el ancho y el profundo fallan los dos, no hay forma de saber si el
  problema es la forma del modelo o que no le alcanza el tamaño: ese es el "ruido de la escala" que TinyStories saca
  del medio. El paper lo encontró en grande: modelos de ~125M de parámetros entrenados sobre corpus generales casi
  nunca llegan a inglés coherente, y modelos de menos de 10M entrenados sobre TinyStories sí.

**El experimento que lo contestaría** es entrenar el mismo modelo ancho con 82M tokens de una muestra web (y un BPE
de 8.192 entrenado sobre esa muestra) y comparar bits por carácter, muestras y notas del juez. No lo hicimos: es otra
corrida entera de las Etapas 1 y 2, y quedó afuera por tiempo (ver el registro de decisiones, abajo).
"""

# %% [markdown]
r"""
## Registro de decisiones del proyecto

Las fechas salen del historial del repositorio y los tiempos, de los logs de las corridas.

### Con qué contábamos al arrancar

- **Gente y tiempo:** cuatro integrantes y tres semanas de calendario (la consigna salió el 6/9 y se entrega el
  27/9). En la práctica, el trabajo que quedó registrado se concentró en los últimos tres días, del 25/9 al 27/9. Ese
  fue el presupuesto real, y explica varias decisiones de abajo: todo tenía que correr en minutos, no en horas.
- **Máquinas:** Colab gratuito (T4 de 16 GB, sesiones que se cortan, disco que se borra), que es el entorno de la
  consigna; una notebook con RTX 5070 Laptop (8 GB de VRAM, 32 GB de RAM, Linux); y otra computadora con Windows.
- **Cuánto rendía una corrida:** lo medimos corriendo todo primero en modo prueba (`SMOKE_TEST`, unos 5 minutos) y
  después la corrida completa: 31 minutos en la RTX 5070 para las Etapas 0 y 2 a 5 (la 2, 17 minutos con los dos
  modelos; la 4, 8; la 5, 6). La Etapa 1 ya estaba corrida y tarda un par de minutos.

### El plan: qué no se recortaba y qué sí

1. **Las Etapas 1 y 2, con la ablación, completas.** La 1 alimenta a todas, y la ablación es lo que convierte esto
   en un laboratorio.
2. **La Etapa 3 completa:** no entrena nada y corre en segundos en CPU, así que el esfuerzo podía ir a los controles
   de ruido.
3. **La Etapa 4 con el SFT completo** (sin él no hay Etapa 5); LoRA, solo si entraba.
4. **La Etapa 5 al final:** si faltaba tiempo, era lo primero en recortarse ("recorten desde el final").
5. **Tamaños elegidos para que la Etapa 2 entrara en minutos**, dentro de los rangos de la §7, y siempre una pasada
   de `SMOKE_TEST` antes de la corrida larga.

### Diario

| Fecha | Qué hicimos | Qué decidimos, y por qué |
|---|---|---|
| 25/9 | Etapas 0 a 2 | `V = 8.192`: el techo del rango de la consigna, cerca de los ~10K del paper; el barrido mostró que más arriba ya casi no se acortan los cuentos (quedó la duda contra 4.096). `block_size = 256`, porque entra el 90 % de los cuentos contra el 60 % con 192. Para la ablación, el par de la §7 con cabezas de 32 dimensiones en los dos modelos, así solo cambian el ancho y la cantidad de capas; contamos los parámetros antes de entrenar (7,2 % de diferencia). 400.000 cuentos y 5.000 pasos: menos de una pasada por el corpus (0,93 épocas) y unos 15 minutos para los dos modelos. |
| 26/9 | Etapa 3 | Pares fijados antes de mirar (*dog–cat*, *happy–sad*, *dog–spoon*), 10.000 pares al azar como control y el máximo de una búsqueda como segundo piso de ruido; una sola PCA ajustada sobre los dos estados, para que los paneles compartan ejes. |
| 26/9 | Etapa 4 | Cada ejemplo de SFT alineado al principio de su registro (no ventanas al azar), para que el modelo nunca vea un cuento sin su instrucción; `lr` 1e-4 y 1.500 pasos, según la §7; LoRA a mano, rango 8, sobre las proyecciones de atención. Medimos una trampa: un prompt que termina en espacio o salto de línea hace que el modelo escupa bytes sueltos, así que los prompts se cortan en `Story:`. |
| 26/9 | Etapa 5 | Anclas para el juez (cuento real, palabras mezcladas, instrucción ajena), porque sin ellas no sabíamos si el juez discrimina; caché de respuestas, para que un corte no obligue a repetir llamadas. |
| 26/9 | Infraestructura | Huella SHA-1 del tokenizer en cada checkpoint: volver a correr la Etapa 1 pisaba el tokenizer y dejaba a los modelos leyendo ids con otro significado, sin ningún error. Un script corre las seis etapas en orden y guarda los notebooks ejecutados. |
| 26/9 | Corrida completa | Primero en modo prueba y después la completa, en la notebook con RTX 5070 en lugar de Colab (ver "Dónde terminamos"). |
| 27/9 | Informe | Un informe por etapa dentro de su notebook, repartidos entre los cuatro; revisión final contra la consigna. |

### Qué se rompió o nos sorprendió, y qué hicimos

- **Un notebook perdió sus salidas** al regenerarlo desde su fuente `.py`: lo restauramos desde la copia ejecutada.
- **El chequeo de GPU podía colgarse** (drivers de GPU híbrida, o un proceso anterior de Python trabado con la GPU
  tomada): lo pasamos a un proceso aparte, con límite de tiempo.
- **`torch.load` se negaba a abrir un checkpoint** del SFT porque el historial guardaba escalares de numpy (desde torch
  2.6 carga con `weights_only=True`): ahora se guardan como `float`.
- **La memoria de GPU de LoRA salía sesgada:** la medición incluía los modelos que ya estaban cargados (en la corrida
  de LoRA, también el de SFT completo). Ahora se mide por encima de lo residente.
- **Texto mal codificado en el corpus** (6 % de los cuentos), que encontramos recién al analizar la Etapa 1.
- **Resultados que había que leer con cuidado:** la cota inferior del intervalo del base en la Etapa 4 es un cero con
  error de redondeo (3,5·10⁻¹⁸), no una ganancia sobre el azar; y el juez, que separa muy bien las anclas, no sirve
  para medir obediencia (efecto halo, Etapa 5).

### Qué recortamos, por qué, y qué nos costó

| Recorte | Por qué | Qué nos costó |
|---|---|---|
| Una sola semilla por configuración | Cada semilla es otra corrida de la Etapa 2 y de todo lo que sigue; preferimos cubrir las cinco etapas | Las diferencias de décimas entre el ancho y el profundo no se pueden separar del ruido |
| 400.000 cuentos y 5.000 pasos (0,93 épocas) | Que la Etapa 2 entrara en unos 15 minutos, casi sin repetir datos | Quedamos en el 70 % de la referencia de Chinchilla contando todos los parámetros, y los dos modelos seguían bajando al cortar |
| No comparar `V = 4.096` contra `8.192` | Exigía repetir la Etapa 2 entera | La duda de la Etapa 1 queda abierta |
| No limpiar el texto mal codificado | Lo vimos cuando el tokenizer ya lo usaban las demás etapas | Los modelos a veces generan `â€œ` |
| SFT sin enmascarar la instrucción ni filtrar a registros con `Words:` | Formato de continuación pelado, como pide la consigna, y un único presupuesto de 1.500 pasos | El contenido se movió a medias (Etapa 4) |
| Pocas muestras: 40 generaciones por celda en la Etapa 4 y 24 textos por modelo y tipo en la 5 | Las fijamos antes de correr, sin estimar cuántas hacían falta. En retrospectiva no era un límite real: el juez evaluó los 264 textos en menos de 3 minutos | Solo se pueden leer las diferencias grandes; con más muestras se habrían podido resolver las de décimas |
| Sin un control con texto web | Otra corrida completa de las Etapas 1 y 2 | La tesis "mejores datos" se apoya en el paper y en el modelo de Shakespeare de la clase, no en un control propio (ver el cierre del `05`) |

### Dónde terminamos, contra el plan

- **Hicimos las cinco etapas**, incluidas las dos partes opcionales (LoRA y el juez). No hizo falta recortar etapas:
  el recorte fue en profundidad dentro de cada una (una semilla, subconjuntos, pocas muestras).
- **El cambio más grande fue dónde corrimos.** La consigna apunta a Colab, y los notebooks siguen preparados para eso
  (primera celda, bloque de Drive), pero la corrida final la hicimos en la notebook con RTX 5070: el pipeline entero
  tardaba poco más de media hora y así no dependíamos de sesiones que se cortan ni de montar Drive para no perder los
  checkpoints. El costo: no medimos cuánto tarda en una T4.
- **Lo que no esperábamos encontrar** terminó siendo de lo más interesante: que el SFT instala la forma mucho antes
  que el contenido, que LoRA casi no ahorra memoria a esta escala, que el juez separa las anclas pero no sabe medir
  obediencia, y que un corpus pensado como "limpio" trae un 6 % de texto mal codificado.
"""

# %% [markdown]
r"""
## Uso de IA

Usamos un asistente de programación (Claude, de Anthropic), y queda a la vista en el historial del repositorio: los
commits que escribió van firmados por él.

- **Código:** escribió la mayor parte del código de las Etapas 4 y 5 y de la infraestructura (el script que corre
  todas las etapas en orden, la huella del tokenizer, la medición de memoria) a partir de lo que le pedimos; nosotros lo corrimos, lo revisamos y
  decidimos qué medir.
- **Resultados y borradores:** ordenó en tablas los números medidos de cada etapa y redactó un primer borrador de los
  resultados de las Etapas 4 y 5.
- **Revisión final (27/9):** revisó los notebooks contra la consigna, chequeó los números citados contra las salidas
  y los CSV, corrigió afirmaciones que habían quedado desactualizadas y redactó, a partir de lo ya medido, la
  ampliación de la Etapa 0 (las respuestas cortas las escribimos nosotros), este registro y el cierre del proyecto. También armó el PDF con las conclusiones.

**Dónde no le creímos.** El resumen preliminar de resultados que armó el asistente comparaba el final de las curvas de
los dos modelos de la ablación en tramos distintos, y atribuía al corpus, sin verificarlo, el `â€œ` que generó uno de
ellos. En el informe de la Etapa 2 los comparamos en el mismo tramo (del paso 4.250 al 5.000: −0,020 el ancho contra
−0,025 el profundo) y tratamos el `â€œ` como hipótesis, hasta que la Etapa 1 midió el texto mal codificado del corpus.
También agregamos lo que el resumen no decía: que repetir un nombre no es recordarlo (en las muestras, Mia termina
huyendo de Mia).
"""
