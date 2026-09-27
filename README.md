# Laboratorio de Lenguaje Emergente · Entrega del grupo

**IA Generativa Avanzada y Sistemas Multi-Agente** — Licenciatura en Tecnología Digital, Universidad Torcuato Di
Tella. Entrega del domingo 27 de septiembre de 2026.

**Grupo:** Facundo Guledjian · Galo Resnik · Lucas Allara · Facundo Riedel

El ciclo de vida de un LLM a escala mínima — tokenizador → preentrenamiento → representación aprendida → SFT →
evaluación con un juez — sobre [TinyStories](https://arxiv.org/abs/2305.07759).

## La entrega

| Archivo | Qué es |
|---|---|
| `00_dataset.ipynb` | Etapa 0: TinyStories como instrumento. Al final, el **registro de decisiones** y el **uso de IA**. |
| `01_tokenizer.ipynb` | Etapa 1: BPE propio a nivel byte (`V = 8.192`) contra GPT-2. |
| `02_pretraining.ipynb` | Etapa 2: el GPT de la clase sobre TinyStories; ablación ancho contra profundo. |
| `03_embeddings.ipynb` | Etapa 3: los embeddings del paso 0 contra los del paso 5.000. |
| `04_sft.ipynb` | Etapa 4: SFT completo y LoRA a mano; forma contra contenido. |
| `05_judge.ipynb` | Etapa 5: juez local (Ollama + `qwen3:4b`) contra perplejidad; cierre del proyecto. |
| `informe_conclusiones.pdf` | Las conclusiones de todas las etapas, el cierre, el registro de decisiones y el uso de IA, en un solo documento. |

Los seis notebooks están **ejecutados**, con sus salidas: corrieron de punta a punta en una notebook con RTX 5070
Laptop (8 GB), con semilla 1337.

## Cómo correrlos en Colab

1. Abrir el notebook en Colab:
   [00](https://colab.research.google.com/github/facurie/Proyecto1_IA_Generativa/blob/main/00_dataset.ipynb) ·
   [01](https://colab.research.google.com/github/facurie/Proyecto1_IA_Generativa/blob/main/01_tokenizer.ipynb) ·
   [02](https://colab.research.google.com/github/facurie/Proyecto1_IA_Generativa/blob/main/02_pretraining.ipynb) ·
   [03](https://colab.research.google.com/github/facurie/Proyecto1_IA_Generativa/blob/main/03_embeddings.ipynb) ·
   [04](https://colab.research.google.com/github/facurie/Proyecto1_IA_Generativa/blob/main/04_sft.ipynb) ·
   [05](https://colab.research.google.com/github/facurie/Proyecto1_IA_Generativa/blob/main/05_judge.ipynb)
   (o subir el `.ipynb` con *Archivo → Subir notebook*).
2. **Antes de correr nada**, elegir la GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4*.
   Sin GPU, las Etapas 2, 4 y 5 pasan solas a `SMOKE_TEST` (una validación corta, sus números no significan nada).
3. Correr la primera celda: clona este repositorio, entra a la carpeta e instala `requirements.txt`. Después,
   *Entorno de ejecución → Ejecutar todas*.

El repositorio trae el tokenizer y los modelos ya entrenados (`checkpoints/`), así que cada notebook corre por su
cuenta, en cualquier orden: las Etapas 2 y 4 cargan los checkpoints en lugar de reentrenar (borrando
`checkpoints/pretrain_*.pt` o `checkpoints/sft_*.pt` se entrena de cero). Las Etapas 0, 1, 2, 4 y 5 bajan
TinyStories de HuggingFace, y la 5 además instala Ollama y baja `qwen3:4b` (~2,5 GB).

## Lo que los notebooks necesitan para correr

| Ruta | Qué es |
|---|---|
| `environment.py` | Cableado del entorno: caché de HuggingFace dentro del repositorio, detección de la GPU, `checkpoints/`. |
| `data.py` | Carga de TinyStories y TinyStories-Instruct (reagrupa las líneas de Instruct en registros completos). |
| `requirements.txt` | Dependencias; la primera celda de cada notebook las instala. |
| `checkpoints/` | Tokenizer, modelos entrenados (y los de `SMOKE_TEST`, en `checkpoints/smoke/`) y las tablas que lee la Etapa 5. |

Todos los modelos están hechos a mano en PyTorch: no se usa `transformers`, `peft`, `bitsandbytes` ni `trl`. El
juez corre local, con Ollama; no hay APIs pagas.

## Créditos y licencia

Parte del código de base (`environment.py`, `data.py`) viene del repositorio de la materia. Licencia MIT — ver
[LICENSE](LICENSE). Copyright (c) 2026 Francisco Traversaro. TinyStories y TinyStories-Instruct se descargan de
HuggingFace al correr los notebooks; no se incluyen en el repositorio.
