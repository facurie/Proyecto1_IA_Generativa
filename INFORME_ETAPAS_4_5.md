# Informe · Etapas 4 y 5

Corrida completa local (RTX 5070 Laptop, 8 GB), `run_all.py --modo completo`, semilla 1337. Base: modelo
**ancho** de la Etapa 2 (`n_embd=256`, `n_layer=2`, 5,85M parámetros, 5.000 pasos, val loss 2,31 / ppl 10,1).
Todas las cifras salen de `checkpoints/stage4/` y `checkpoints/stage5/`. Hay una sola semilla por
configuración: los intervalos de confianza reflejan la variación entre prompts, no entre corridas.

---

## Etapa 4 · SFT: comportamiento, no conocimiento

### Qué se entrenó

- **SFT completo:** 1.500 pasos, batch 64, `lr` 1e-4 → 1e-5, sobre 100.000 registros de
  TinyStories-Instruct, cada uno alineado al principio (instrucción + `Story:` + cuento). Tardó 1,7 min.
- **SFT LoRA:** mismo presupuesto, `lr` 1e-3, rango 8 y α=16 sobre `Head.key/query/value` y
  `MultiHeadAttention.proj`. El resto del modelo quedó congelado.
- **Medición:** 8 prompts de 3 palabras × 5 muestras, en dos plantillas:
  - `Words + Story`: `Words: …` + `Story:`. Mide sobre todo el contenido.
  - `solo Words`: solamente la línea `Words: …`, y el modelo tiene que saber que después viene un cuento.
    Mide sobre todo la forma.

### El par antes/después: `Words: dragon, happy, forest`

Misma semilla para los tres modelos. Fragmentos textuales:

**Plantilla `solo Words`** (el prompt termina en `forest`):

> **Base:** `, and it would be so loud that day he was always bigger than the other animals were talking. One day Sam's parents said to him, "Let's go and see where the other person can find my secret from here!" …`

> **SFT completo:** `, good`
> `Summary: Tim learns to share a bigger map by making his friend happy ending.`
> `Story: Once upon a time, in a big forest, there was a green forest. On one last, he met a small rabbit. The rabbit felt happy and grateful. …`

> **SFT LoRA:** `, good` / `Summary: Tim observes a game where he needed food for the dragon …` / `Summary: …` / `Dort: Timmy remembers …` (no llega a escribir `Story:`)

**Plantilla `Words + Story`:**

> **Base:** `"Mr Jones! Are you so big? What's wrong?" Mama: "I'll bring to the dragon inside the candle". Slowly, the dragon kept walking away …`, usa 1 de 3 palabras.

> **SFT completo:** `Once upon a time, there was a little bear named Benny. Benny loved to hop around the forest near his home. … Benny was happy …`, usa 2 de 3.

> **SFT LoRA:** `Lily loved to hide with her friend, Timmy. …`, usa 1 de 3 (`happy`).

### ¿Qué cambió: la forma o el contenido? Las dos, pero no por igual

| | arranca cuento (`solo Words`) | termina (`W+S`) | uso de palabras (`W+S`) | piso de azar | uso − piso [IC95] |
|---|---|---|---|---|---|
| base | **0 %** | 92,5 % | 19 % | 8 % | 0,11 [0,00, 0,19] |
| SFT completo | **92,5 %** | 92,5 % | **52,5 %** | 21 % | **0,32 [0,14, 0,46]** |
| SFT LoRA | 37,5 % | 85 % | 36 % | 11 % | 0,25 [0,17, 0,33] |

- **La forma cambió de golpe y por completo.** Con solo la línea `Words:`, el base nunca escribió un
  cuento: sigue la frase como si `forest` fuera parte de una oración (`, and it would be so loud…`). El
  SFT completo, en el 92,5 % de los casos, trata esa línea como lo que es, un encabezado. Primero completa
  la lista (`, good`), después agrega otro encabezado (`Summary:`), escribe `Story:` y arranca el cuento.
  Eso no es conocimiento nuevo sobre dragones o bosques: es haber aprendido la estructura del formato.
  En la medición chica durante el entrenamiento, "arranca cuento" ya estaba en 75 % en el paso 300 (de
  1.500) y en 100 % desde el 900.
- **En `Words + Story` la forma ya estaba antes.** El base arranca un cuento el 92,5 % de las veces
  cuando el prompt termina en `Story:`, porque el preentrenamiento ya le enseñó a escribir cuentos. Lo que
  el SFT agregó en esa plantilla es contenido, no forma.
- **El contenido se movió, pero a medias.** El uso de las palabras pedidas pasó de 19 % a 52,5 %. La
  ganancia sobre el piso de azar (0,32) casi triplica la del base (0,11), y el intervalo del base toca el
  cero: el base solo "usa" palabras frecuentes en TinyStories (`happy`, `forest`) por casualidad. Aun así,
  el SFT completo usa en promedio la mitad de las palabras, no las tres. Durante el entrenamiento el uso
  osciló entre 0,33 y 0,75 sin una tendencia limpia, mientras que la forma subió de manera monótona.
- **Costo en olvido.** La pérdida en TinyStories liso subió de 2,29 a 2,47 con el completo y a 2,52 con
  LoRA. En la Etapa 5, con otros batches, dio 2,33 → 2,57 y 2,54. El olvido existe y es moderado. El orden
  entre completo y LoRA se invierte según la medición, así que la diferencia entre ellos está dentro del
  ruido.
- **Efecto secundario.** El SFT a veces mete líneas de encabezado dentro del cuento: 0,6 por generación
  en el completo, 0 en el base. Aprendió tan bien el formato que a veces lo sobreaplica.

**Conclusión.** El SFT movió primero y del todo la forma: reconoce la instrucción y la convierte en un
cuento. El contenido se movió menos: usa más de las palabras pedidas que el azar, pero no todas.

**Qué haríamos para conseguir el resto del contenido:**
1. **Enmascarar la pérdida de la instrucción**, para que todo el gradiente vaya al cuento condicionado.
   Hoy el modelo también aprende a generar encabezados, y eso explica el `, good` y los `Summary:`
   espontáneos.
2. **Filtrar a registros con `Words:`.** Solo una parte de Instruct tiene esa línea, así que la señal de
   "usá estas palabras" es más rala que la de formato, que está en todos los registros.
3. **Más pasos:** el uso de palabras no había convergido.
4. **Un modelo más profundo.** Usar una palabra 50-100 tokens después de leerla es un problema de
   contexto, y con 2 capas hay poco margen.

### LoRA: qué fracción entrenamos y cuánto de la mejora compró

| | completo | LoRA |
|---|---|---|
| parámetros entrenables | 5.846.528 (100 %) | **118.784 (1,99 %)** |
| gradientes + AdamW (analítico) | 70,2 MB | 1,4 MB |
| memoria pico GPU al entrenar | 3.065 MB | 2.922 MB (**−4,6 %**) |
| segundos por paso | 0,067 | 0,062 (−7 %) |

**Cuánto de la mejora compró LoRA**, medido como `(LoRA − base) / (completo − base)`:

| métrica | base | completo | LoRA | LoRA / completo |
|---|---|---|---|---|
| pérdida en Instruct (val) | 3,20 | 2,12 | 2,35 | **78 %** |
| contenido: uso − piso | 0,11 | 0,32 | 0,25 | **68 %** |
| forma: arranca cuento (`solo Words`) | 0 % | 92,5 % | 37,5 % | **41 %** |

- **Con el 2 % de los parámetros, LoRA compró el 78 % de la mejora en pérdida** y el 68 % de la de
  contenido. En comportamiento visible compró mucho menos: el 41 % de la forma.
- **La forma es lo que más le cuesta a LoRA, y tiene sentido.** Para escribir `Story:` en el momento
  justo hay que cambiar qué token es probable en un punto dado, y eso vive en el MLP y en `lm_head`, que
  LoRA dejó congelados. LoRA solo puede cambiar cómo se mira el contexto. En el par de arriba se ve: LoRA
  aprendió a producir encabezados (`Summary:`), pero se queda dando vueltas en ellos sin llegar a `Story:`.
  Como variantes para probar: aplicar LoRA también al MLP, o descongelar `lm_head`.
- **La memoria casi no bajó (−4,6 %).** Los gradientes y el estado de AdamW del modelo completo ocupan
  70 MB, y el pico pasa los 3 GB. A esta escala dominan las activaciones, sobre todo los logits:
  64 × 256 × 8.192 posiciones por batch, en fp16 más su copia en fp32 para la pérdida. LoRA achica
  exactamente la parte que acá no pesa. Su ventaja de memoria aparece en modelos donde los pesos y el
  optimizador dominan el pico, no en uno de 6M de parámetros.

---

## Etapa 5 · El juez local (qwen3:4b, `think=False`, `format="json"`)

264 evaluaciones:
- 4 modelos × 48 textos, cada uno con 6 prompts de cuento y 6 de instrucción × 4 muestras;
- 72 anclas: cuentos reales, esos mismos cuentos con las palabras mezcladas, y cuentos reales evaluados
  contra una instrucción que no les corresponde.

### ¿Es confiable el juez?

- **Formato: 264 de 264 respuestas válidas**, ninguna con texto extra ni con dos objetos. Con
  `format="json"`, la trampa de "el juez contesta dos veces" no apareció en esta corrida. El parser
  robusto quedó como seguro, sin llegar a usarse.
- **Separa bien las anclas:**

  | ancla | gramática | creatividad | consistencia |
  |---|---|---|---|
  | cuento real | 8,3 [8,0, 8,5] | 6,8 [6,5, 7,1] | 8,8 [8,6, 9,0] |
  | palabras mezcladas | 1,5 [1,3, 1,8] | 2,5 [2,2, 2,7] | 1,5 [1,3, 1,8] |

  Mismo vocabulario y misma temática, con la gramática destruida: la nota cae de 8 a 1,5. El juez lee
  gramática y coherencia, no solo el tema.
- **La obediencia no es confiable.** Los cuentos reales evaluados contra una instrucción ajena
  (12,5 % de uso de las palabras pedidas) sacaron obediencia 5,75. Los del SFT completo sacaron 4,0 tanto
  con 0 como con 3 de 3 palabras usadas. La correlación entre la obediencia del juez y el uso real de
  palabras es de apenas 0,23 (Spearman). La obediencia del juez está contaminada por la calidad general
  del texto (efecto halo). Para obediencia, la cuenta automática de la Etapa 4 es mejor instrumento.

### Notas por modelo (prompts de cuento; en obediencia, prompts de instrucción)

| | val ppl (liso) | gramática | creatividad | consistencia | obediencia | uso de palabras |
|---|---|---|---|---|---|---|
| base (ancho) | **10,3** | 1,96 | 3,08 | 2,17 [2,0, 2,4] | 2,8 [2,3, 3,3] | 18 % |
| profundo | 10,6 | 1,92 | 2,92 | 1,96 [1,9, 2,0] | 2,5 [2,0, 3,0] | 21 % |
| SFT completo | 13,1 | 2,00 | 2,96 | 1,88 | **4,1 [3,9, 4,4]** | **56 %** |
| SFT LoRA | 12,7 | 1,96 | 2,92 | 1,83 | 3,7 [3,2, 4,0] | 33 % |

**Hay un efecto piso.** De las 192 generaciones de nuestros modelos, el juez puso gramática 2 en 173 y
1 en 19, nunca más. Contra la escala de un cuento real (8,3), los cuatro modelos están en el mismo
escalón. El juez no puede ordenarlos en gramática. Las diferencias en consistencia son de décimas y, salvo
base contra el resto, los intervalos se pisan.

### ¿Dónde le da la razón el juez a la perplejidad, y dónde se le va para otro lado?

**Coinciden:**
- **Texto por texto, dentro de nuestros modelos:** cuanto menor la NLL de un texto (medida con el modelo
  ancho), mejor la nota. Spearman: gramática −0,35, creatividad −0,37, consistencia −0,44.
- **Base contra profundo:** la perplejidad prefiere al ancho (10,3 contra 10,6) y el juez también, por
  poco (consistencia 2,17 contra 1,96).
- **Contra la Etapa 2:** esa preferencia coincide con la probabilidad media de la respuesta correcta en
  los prompts de recuerdo (0,12 contra 0,10). No coincide con el reuso de palabras clave, que favorecía al
  profundo (0,58 contra 0,52). Ninguna de las tres lecturas reproduce "profundidad ↔ contexto" del paper a
  esta escala.

**Se separan, en dos lugares que dicen mucho de qué mide la perplejidad:**

1. **El SFT tiene la peor perplejidad y el juez no lo castiga.**
   - En texto liso, el SFT completo sube de 10,3 a 13,1 de perplejidad: es el peor de los cuatro.
   - El juez no lo ve peor: gramática igual (2,0) y la obediencia más alta (4,1 contra 2,8).
   - La perplejidad sobre TinyStories mide cuánto se parece el modelo a esa distribución. El SFT se
     corrió hacia otro formato, y eso se paga en perplejidad aunque los cuentos no empeoren.
   - En su propia distribución (Instruct), el SFT baja de 3,20 a 2,12. La perplejidad depende de sobre
     qué texto se mide; la calidad no.
2. **Un cuento real puntúa casi igual que el balbuceo del propio modelo.**
   - Con el modelo ancho como vara, los cuentos reales tienen una NLL de 2,39 por token, y las
     generaciones del propio ancho, 2,29: le parecen más probables sus propias muestras que los cuentos
     humanos.
   - El juez las separa por 6 puntos (gramática 8,3 contra 1,96).
   - La perplejidad bajo un modelo mide cuánto se parece un texto a lo que ese modelo produciría, no si
     es bueno. Por eso el propio modelo se autoevalúa bien.
   - Solo el caso extremo de las palabras mezcladas (NLL 8,8) lo separan las dos varas por igual.

**Conclusión.** La perplejidad es buena para detectar texto roto y para comparar modelos sobre la misma
distribución. No sirve para decidir si un texto es un buen cuento, y penaliza cualquier cambio de formato
como si fuera pérdida de calidad. El juez sirve justo para eso, con dos límites medidos: satura contra un
piso cuando todos los modelos son malos, y su obediencia se deja llevar por la calidad general del texto.
Para seguir instrucciones, la medida automática (uso − piso) resultó más confiable que el juez.

---

## Limitaciones

- Una sola semilla de entrenamiento por configuración: no hay variabilidad entre corridas.
- En la Etapa 4, 40 generaciones por celda; en la Etapa 5, 24 textos por modelo y tipo. Alcanza para las
  diferencias grandes (forma, obediencia del SFT), no para las de décimas.
- El juez tiene 4B parámetros. Con un juez más grande, el efecto piso probablemente se abriría en más
  escalones.
