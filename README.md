# Laboratorio de Lenguaje Emergente

<!-- ENCABEZADO POR INSTANCIA — editá estas dos líneas cuando reutilices esto en otra materia. -->

**IA Generativa Avanzada y Sistemas Multi-Agente** — Licenciatura en Tecnología Digital, Universidad
Torcuato Di Tella.

Un proyecto de la materia, con nota. Reproduce, a escala mínima y enteramente en
Colab, el ciclo de vida real de un LLM — tokenizador → preentrenamiento → representación aprendida → SFT — sobre
[TinyStories](https://arxiv.org/abs/2305.07759), y pregunta *por qué* el lenguaje "emerge" cuando el trabajo lo hace
el dato y no la escala.

## Empezá por acá

**[assignment.md](assignment.md)** — la consigna completa: cinco etapas, paso a paso, con
fragmentos de código adaptables y las preguntas conceptuales que componen la nota.

## Preparación del entorno

### En Colab (el entorno de ejecución objetivo)

No hay nada que instalar, pero el orden importa:

1. **Elegí primero el entorno de ejecución** — *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4*. Cambiar el tipo de entorno
   reemplaza la VM y borra `/content`, así que hacerlo después tira a la basura todo lo que hizo el paso 2.
2. **Ejecutá la primera celda.** Clona el repositorio del grupo, hace `cd` a él e instala `requirements.txt`.
   Todos los imports posteriores resuelven; nada funciona antes de eso.

Si en algún momento ves `ModuleNotFoundError: No module named 'environment'`, el entorno de ejecución se recicló —
volvé a ejecutar la primera celda.

> El disco de Colab se borra cuando el entorno de ejecución se desconecta, y las Etapas 3–5 necesitan todas los
> checkpoints de la Etapa 2. La celda de arranque tiene un bloque comentado para montar Drive — descomentalo, o descargá
> tus checkpoints antes de cerrar la pestaña.

### Localmente (para correr `SMOKE_TEST` en CPU)

```bash
git clone https://github.com/facurie/Proyecto1_IA_Generativa.git
cd Proyecto1_IA_Generativa

python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows (PowerShell)
# source .venv/Scripts/activate     # Windows (Git Bash)
# source .venv/bin/activate         # Linux / macOS

pip install -r requirements.txt
python verify_setup.py
```

`verify_setup.py` chequea tu versión de Python, las dependencias y la accesibilidad del HuggingFace Hub
(una llamada chica de metadatos, no una descarga completa). Después empezá con `00_dataset`.

Ejecutá los archivos de cada etapa desde la raíz del repositorio — leen y escriben `checkpoints/` por ruta relativa.

## Qué hay acá

| Ruta | Qué es |
|---|---|
| [assignment.md](assignment.md) | **La consigna.** |
| `00_dataset.py` | Sección 0 — descargar TinyStories, inspeccionarlo, argumentar por qué un vocabulario acotado es un instrumento de laboratorio. |
| `data.py` | Cargadores de datasets compartidos — incluye un parser hecho desde cero para el formato de una-línea-por-fila de `TinyStories-Instruct`. |
| `environment.py` | Cableado del entorno local: caché de HF dentro del repositorio, bundle de certificados para proxy corporativo, detección del dispositivo. |
| `tools/py_to_notebook.py` | Regenera el `.ipynb` de cada etapa a partir de su fuente `.py`, conservando las salidas de las celdas cuyo código no cambió. |
| `tools/ca_bundle_windows.py` | Corrige la falla de TLS de `huggingface_hub` detrás de un proxy corporativo que inspecciona TLS. |
| `verify_setup.py` | Chequeo del entorno. |

## Notas de diseño

- **Sin `transformers`/`peft`/`bitsandbytes`/`trl`.** Todos los modelos de este pipeline — el Transformer de
  preentrenamiento, la continuación con SFT, los adaptadores LoRA opcionales — están hechos a mano en PyTorch puro,
  reutilizando el código del Transformer desde cero que ya escribiste. No hay ningún modelo de HuggingFace al que
  engancharle un adaptador de librería, y cuantizar un modelo tan chico no ahorra nada.
- **Solo código abierto, sin APIs pagas.** El juez (Etapa 5, opcional) corre un modelo local a través de Ollama.
- **El entorno objetivo es Colab T4.** Este repositorio también se desarrolló en una máquina sin GPU, así que dale a cada
  etapa de entrenamiento que escribas un flag `SMOKE_TEST` que valide la lógica sobre un subconjunto mínimo en minutos,
  sin GPU — `00_dataset.py` muestra el patrón. La Etapa 5 es la excepción: el juez necesita
  Ollama y está pensado para correr dentro de Colab.

## Qué escribís vos

Este repositorio te da los cargadores de datasets, el cableado del entorno y la Sección 0. **Las Etapas 1–5 son
tuyas para construir** — incluido el Transformer mismo, que reutilizás de tu propio trabajo anterior
en lugar de importarlo de acá. [assignment.md](assignment.md) tiene fragmentos adaptables para cada
etapa y, en la §3b, los nombres de archivo con los que las etapas se pasan artefactos entre sí.

## Créditos y licencia

Reutiliza y extiende material previo de la materia sobre Transformers desde cero, tokenizadores, embeddings
y fine-tuning. TinyStories y TinyStories-Instruct se descargan de HuggingFace en tiempo de ejecución,
nunca se incluyen en el repositorio.

Licencia MIT — ver [LICENSE](LICENSE). Copyright (c) 2026 Francisco Traversaro.
