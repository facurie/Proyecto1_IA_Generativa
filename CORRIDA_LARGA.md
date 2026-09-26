# Corrida larga: de punta a punta en Colab

Checklist para la corrida final, con todas las etapas a escala completa en una T4. Cada notebook se
puede cortar y volver a correr: las Etapas 2 y 4 retoman o cargan lo que ya está entrenado, y el juez de la
Etapa 5 guarda cada respuesta a medida que llega.

## Antes de empezar

1. **GPU primero:** Entorno de ejecución → Cambiar tipo de entorno de ejecución → **T4 GPU**. Cambiar el
   tipo de entorno después borra todo.
2. **`checkpoints/` en Drive, en todos los notebooks.** En la primera celda de cada notebook (la de
   arranque) descomenten el bloque de Drive, con la **misma** ruta `PERSISTENT` en todos. Sin eso, una
   desconexión (≈90 min de inactividad, 12 h como máximo) se lleva todos los modelos.
3. **No vuelvan a correr la Etapa 1 si quieren conservar los modelos entrenados.** `01_tokenizer` pisa
   `tokenizer.json` siempre, y un modelo entrenado con otro tokenizer carga sin error pero lee ids que ya no
   significan lo mismo. Desde ahora cada checkpoint guarda la huella (`tokenizer_sha1`) del tokenizer con
   el que se entrenó, y las Etapas 2, 4 y 5 se frenan con un error si no coincide. Los checkpoints
   anteriores a este cambio no tienen huella: se aceptan, pero sin esa garantía.

## Orden y qué deja cada etapa

| Notebook | Lee | Escribe en `checkpoints/` | Notas |
|---|---|---|---|
| `00_dataset` | — | `*.png` | Opcional en la corrida final. |
| `01_tokenizer` | — | `tokenizer.json` | Ver el punto 3 de arriba. |
| `02_pretraining` | `tokenizer.json` | `pretrain_step0.pt`, `pretrain_wide.pt`, `pretrain_deep.pt`, `stage2_ablation.csv`, `stage2_curves.png` | La más larga: dos modelos × 5.000 pasos. Retoma sola si se corta. |
| `03_embeddings` | `tokenizer.json`, `pretrain_step0.pt`, `pretrain_wide.pt` | `stage3/` | Solo CPU, sin entrenar. |
| `04_sft` | `tokenizer.json`, `pretrain_wide.pt` | `sft_final.pt`, `sft_lora.pt`, `stage4/` | SFT completo y LoRA, 1.500 pasos cada uno. La primera carga de TinyStories-Instruct arma el split entero (~21,7M líneas). |
| `05_judge` | todo lo anterior | `stage5/` | Instala Ollama y baja `qwen3:4b` (~2,5 GB). Unas 260 llamadas al juez. |

Para ver cómo vamos, en cualquier momento:

```python
!ls -la checkpoints/ checkpoints/stage4 checkpoints/stage5
```

## Al terminar

1. **Descargar cada notebook con sus salidas:** Archivo → Descargar → Descargar .ipynb. La entrega pide
   los notebooks ejecutados.
2. **Bajar las tablas y los gráficos** (sin los `.pt`, que no van en la entrega):

   ```python
   !cd checkpoints && zip -r resultados.zip stage2_ablation.csv stage2_curves.png stage3 stage4 stage5
   from google.colab import files
   files.download("checkpoints/resultados.zip")
   ```

3. Con eso se escriben las respuestas de las secciones "Para el informe" de cada notebook.

## Si algo falla

- `FileNotFoundError` de un checkpoint: falta correr la etapa anterior, o Drive no está montado en esta
  sesión (ver el punto 2).
- `ValueError: ... se entrenó con otro tokenizer.json`: se volvió a correr la Etapa 1. Traigan el
  `tokenizer.json` original o reentrenen desde la Etapa 2.
- `... es de otra configuración`: cambiaron hiperparámetros de la Etapa 2 con un checkpoint viejo en
  `checkpoints/`. Bórrenlo o renómbrenlo.
- El juez devuelve respuestas rotas: no es un error, queda contado en `stage5/formato_juez.csv` y es un
  resultado para el informe.
