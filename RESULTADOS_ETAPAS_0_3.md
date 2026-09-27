# Resultados medidos · Etapas 0 a 3 (insumo para el informe)

Este archivo junta **los números y las observaciones verificables** de la corrida completa (RTX 5070
Laptop, semilla 1337), organizados por pregunta de la consigna. No es el informe: la consigna pide que
la interpretación y la opinión las escriban ustedes (§2, "Sobre usar IA"). Cada pregunta termina con lo
que queda por decidir o argumentar.

Fuentes: los notebooks ejecutados en `outputs/notebooks/`, y `checkpoints/stage2_ablation.csv` y
`checkpoints/stage3/*.csv`.

---

## Etapa 0 · El dataset como instrumento

**Qué se midió** (inspección chica: 1.000 cuentos, modo por defecto del notebook):

- 183,8 palabras por cuento en promedio (mínimo 61, máximo 837).
- ~4.961 palabras únicas en esos 1.000 cuentos.
- Las más comunes: the, and, to, a, was, she, he, they, it, her.
- Formato de Instruct confirmado: un subconjunto de `Features:`/`Words:`/`Summary:`, en orden variable,
  después `Story:` y el cuento.

**Datos para responder "¿qué da el vocabulario de un chico de 3-4 años?":**

- Los mismos tokens se reparten sobre muy pocos tipos distintos. En la Etapa 1, el 73 % del vocabulario
  BPE de 8.192 terminó siendo palabras enteras (5.966 tokens `Ġ`+letras), y se hablan ~1,23 tokens por
  palabra.
- Con 5,8M de parámetros, el modelo llega a cuentos con personajes y diálogo en ~2.000 pasos. Ver las
  muestras de la Etapa 2.

**Para escribir ustedes:** qué esperan que cambie con texto web crudo y la misma cantidad de tokens
(vocabulario, distribución de temas, cuánto del modelo se va en tokens raros), y por qué.

---

## Etapa 1 · Tokenizador BPE

**Qué se midió:**

- **Entrenamiento:** V = 8.192, sobre 100.000 cuentos, en 10 s. Ida y vuelta exacta en 5.000 de 5.000
  cuentos held-out.
- **Primeros merges:** `h+e`, `Ġ+t`, `Ġ+a`, `Ġ+s`, `Ġ+w`, `n+d`, `Ġt+he → Ġthe` (el 7.º),
  `Ġa+nd → Ġand` (el 9.º), `Ġt+o`, …, `Ġwa+s → Ġwas` (el 24.º). Casi todos empiezan con `Ġ` (comienzo de
  palabra). Las primeras palabras enteras que aparecen son `the`, `and`, `to`, `was`, las mismas que
  encabezan la frecuencia de la Etapa 0.
- **Tokens más largos:** `Ġaccomplishment`, `Ġgranddaughter`, `Ġunderstanding`, `ĠUnfortunately`,
  `Ġcompassionate`, …

**Comparación sobre los mismos 5.000 cuentos held-out:**

| tokenizador | V | tokens/cuento (media) | mediana | tokens/palabra | params emb + salida (n_embd=256) |
|---|---|---|---|---|---|
| BPE propio | 8.192 | 195,8 | 181 | 1,23 | 4,19M |
| GPT-2 | 50.257 | 199,1 | 184 | 1,26 | 25,73M |
| cl100k (GPT-3.5/4) | 100.277 | 190,9 | 177 | 1,20 | 51,34M |

El BPE propio, con 6 veces menos vocabulario que GPT-2, produce **1,7 % menos** tokens por cuento, y
cl100k apenas un 2,5 % menos que el propio.

**Barrido de V:**

| V | tokens/cuento | params emb + salida | % del vocabulario sin usar en held-out |
|---|---|---|---|
| 512 | 348,3 | 0,26M | 31,6 % |
| 1.024 | 263,9 | 0,52M | 16,5 % |
| 2.048 | 223,5 | 1,05M | 9,4 % |
| 4.096 | 204,3 | 2,10M | 9,1 % |
| 8.192 | 195,9 | 4,19M | 16,5 % |
| 16.384 | 194,4 | 8,39M | 46,1 % |

Largos con este tokenizador: mediana de 182 tokens por cuento, p90 255, p99 544. Entran enteros en la
ventana el 6,1 % de los cuentos con `block_size` 128, el 60,4 % con 192 y el 90,2 % con 256 (se eligió
256).

**Para escribir ustedes:**
- Qué dicen los primeros merges sobre el corpus.
- Si un V más chico es siempre mejor. Datos útiles:
  - De 4.096 a 8.192, los tokens por cuento bajan solo un 4 % y los parámetros de embeddings y salida
    se duplican, del 36 % al 72 % del modelo ancho.
  - A partir de 8.192, la fracción del vocabulario sin usar vuelve a subir.
- Por qué se quedaron con 8.192.

---

## Etapa 2 · Preentrenamiento y ablación ancho/profundo

**Configuración:**
- 400.000 cuentos (88,5M tokens), 5.000 pasos × batch 64 × 256 = 81,9M tokens vistos (0,93 épocas).
- AdamW, `lr` 3e-4 con coseno, mixed precision.
- Tiempos: 6,3 min el ancho y 8,5 min el profundo.

**Muestras con la misma semilla, prompt `Once upon a time` (modelo ancho):**

| paso | val loss | cómo se ve |
|---|---|---|
| 0 | 9,06 (≈ ln 8.192 = 9,01) | tokens sueltos sin relación: `timeronaut mesmerdy children thornale…` |
| 250 | 3,90 | aparece la apertura `Once upon a time, there…` y frases cortas, sin oraciones completas |
| 500 | 3,29 | oraciones cortas, algunas bien formadas (`She was happy.`) |
| 1.000 | 2,86 | párrafos con estructura de cuento; persisten errores (`a little girl named she loved…`) |
| 2.000 | 2,56 | personaje con nombre sostenido (Lily), cierre con moraleja (`From then on…`) |
| 3.500–5.000 | 2,37–2,31 | diálogo con comillas; la gramática local es buena y la coherencia de objetos, no (`she put it on harder`) |

**Observación sobre la semilla fija.** Las mismas palabras raras aparecen en muestras de pasos distintos
e incluso de modelos distintos: `harder`, `tutor`, `amazement`, `peanut`, `stuffed`, `trunk… dirt`.
Aparecen desde el paso 0 hasta el 5.000, en el profundo, y en los modelos de SFT de la Etapa 4. Como el
muestreo usa los mismos números al azar, cuando dos distribuciones se parecen, esos números eligen los
mismos tokens poco probables. La semilla fija sirve para que la diferencia entre muestras venga del
modelo, pero también deja una "huella" compartida en el texto. Eso conviene tenerlo en cuenta al comparar
muestras de a una.

**Otra observación.** El modelo ancho generó `â€œ` en lugar de comillas (`â€œWhat is that?!"`). Es texto
con mala codificación que ya está en TinyStories, y el modelo lo aprendió como cualquier otro patrón.

**Tabla de la ablación:**

| | ancho | profundo |
|---|---|---|
| n_embd / n_layer | 256 / 2 | 192 / 5 |
| parámetros | 5,85M | 5,42M (−7,2 %) |
| parámetros de los bloques (sin embeddings ni salida) | 1,58M | 2,22M (+41 %) |
| train / val loss | 2,273 / **2,311** | 2,297 / 2,336 |
| val ppl | **10,08** | 10,34 |
| recuerdo: aciertos greedy (10 prompts) | 2/10 | 2/10 |
| recuerdo: P(respuesta) media | **0,121** | 0,101 |
| consistencia: reuso de palabras clave | 0,52 | **0,58** |
| juez (Etapa 5), consistencia | 2,17 [2,0, 2,4] | 1,96 [1,9, 2,0] |

Detalles que sirven para leer la tabla:
- **Recuerdo:** cada modelo acierta 2 de 10, pero no los mismos. El ancho acierta `fish → sea` y
  `fire → hot`; el profundo, `fish → pond` y `Birds can → fly`. Los errores típicos siguen el cuento en
  vez del dato: `The sky is → a little girl`, `Bees make → a big tower`.
- **Consistencia:** el reuso se calcula sobre 5 prompts × 5 muestras × 2 palabras = 50 menciones
  posibles. 0,58 contra 0,52 son unas 3 menciones de diferencia.
- **Final de las curvas:** el profundo bajó 0,025 de val loss entre los pasos 4.250 y 5.000, y el ancho
  0,012 entre los pasos 4.500 y 5.000. El profundo seguía bajando algo más rápido al cortar.

**Chinchilla (≈ 20 tokens por parámetro):**

| | N total | tokens vistos / 20·N | N sin embeddings ni lm_head | tokens vistos / 20·N |
|---|---|---|---|---|
| ancho | 5,85M | 0,70 | 1,58M | 2,60 |
| profundo | 5,42M | 0,76 | 2,22M | 1,84 |

**Para escribir ustedes:**
- En qué rango de pérdida aparece estructura de oración (la tabla de muestras da los puntos de corte).
- Si su par reproduce el paper. En la dirección, recuerdo favorece al ancho y reuso al profundo. En el
  tamaño, las diferencias entran en el ruido de 10 y 25 prompts, y el juez contradice al reuso.
- Qué explicación prefieren, y qué medirían con más: más semillas, más prompts de recuerdo, entrenar
  hasta convergencia.
- Qué les costó quedar en 0,70 de Chinchilla contando todos los parámetros, pero 2,6 veces por encima
  contando solo los de los bloques.

---

## Etapa 3 · La representación aprendida (modelo ancho, paso 0 contra paso 5.000)

**Piso de ruido:**
- En 256 dimensiones, `1/√d` = 0,0625.
- El máximo coseno inicial entre 5.965 candidatos tiene una mediana de 0,229 y un p95 de 0,266.
- En los pares al azar de control, el coseno entrenado tiene p95 0,208 y p99 0,304, y el cambio
  (entrenado − inicial) tiene mediana 0,013.

**Pares elegidos antes de correr:**

| par | expectativa | coseno inicial | coseno entrenado | percentil del cambio contra los controles |
|---|---|---|---|---|
| dog–cat | cerca | 0,031 | **0,587** | 99,99 |
| happy–sad | cerca | 0,043 | **0,456** | 99,89 |
| dog–spoon | lejos | −0,019 | 0,157 | 91,6 (por debajo del p95 de los controles) |

**Diez vecinos más cercanos (palabras enteras):**

| palabra | paso 0 | paso 5.000 |
|---|---|---|
| dog | germs 0,22, danger, stay, dogs, Bu… | **cat 0,58, puppy 0,55, pup, mule, wolf, squirrel, doggy**, monster, kitten, Rex |
| happy | breath 0,26, steak, knot, eventually… | **glad 0,52, relieved, excited, lucky, delighted, overjoyed, sad**, happier, pleased |
| sad | everywhere 0,24, fear, Like, frosting… | **upset 0,65, frustrated, miserable, troubled, guilty, unhappy, lonely** |
| spoon | afternoon 0,23, birthday, decor… | **fork 0,51, pan, wand, cup, bowl, plate, pen, knife** |

En el paso 0, los vecinos tienen cosenos de 0,18 a 0,26, justo en el rango del piso de ruido (máximo
típico de 0,23): son ruido puro.

**Proyección 2D:** con PCA, la PC1 explica el 3,55 % de la varianza y la PC2 el 2,05 %, un 5,6 % en
total. El 94 % de la estructura queda fuera del gráfico.

**Filtro de palabras enteras:**
- **Sin el filtro, paso 0:** entre el 60 % y el 90 % de los vecinos son palabras enteras. El resto son
  pedazos: `ving`, `Tweet`, `ee` para dog; `onies`, `sm`, `ache` para cat.
- **Sin el filtro, paso 5.000:** el 100 % de los vecinos de las 5 consultas son palabras enteras, sin
  necesidad de filtrar.
- **El filtro `Ġ`+letras no garantiza palabras enteras:** deja pasar comienzos de palabra como `ĠBu`,
  `Ġru`, `Ġequ` (aparecen entre los vecinos iniciales de dog).
- **`Ġday` (id 356) y `day` (id 1112) son filas distintas:** su coseno entre sí es −0,14 en el paso 0 y
  0,16 después de entrenar. Si se busca por el string pelado, se consultan vecinos de otra fila.

**Para escribir ustedes:**
- Si le pegaron a los pares y si la diferencia supera el ruido: dog–cat y happy–sad superan tanto el
  piso como los controles; dog–spoon, no.
- Qué dice que happy y sad, que son antónimos, queden cerca.
- Qué ven en `pca_comparada.png` y cuánto confiar en una figura que muestra el 5,6 % de la varianza.
- El efecto del filtro.
- Si el espacio entrenado se parece al inicial. Los controles al azar casi no se movieron (cambio mediano
  0,013) y los pares relacionados sí: el cambio es selectivo.

---

## Registro de decisiones y uso de IA

Quedaron escritos dentro de la entrega, al final del notebook `00_dataset` (registro de decisiones del
proyecto y uso de IA); el cierre del proyecto está al final de `05_judge`. Este archivo queda como insumo.
