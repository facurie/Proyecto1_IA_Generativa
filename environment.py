"""
Cableado del entorno local. Importá esto ANTES que tokenizers/datasets/torch.

    import environment            # noqa: F401  (efectos colaterales)
    environment.summary()

QUÉ HACE ESTO, Y POR QUÉ
------------------------
1. Apunta la caché de HuggingFace DENTRO del repositorio (`.hf_cache/`). Por defecto HF
   escribe en `~/.cache/huggingface`, que en Windows vive en el perfil del
   usuario y se llena sin que nadie lo note. Dentro del repositorio la podés
   borrar cuando quieras y ver cuánto ocupa.

2. Conecta un bundle de certificados local si hay uno presente
   (`.certs/ca_bundle.pem`). En una red que inspecciona TLS (muchas redes corporativas y
   algunas de campus), `huggingface_hub` falla con
   CERTIFICATE_VERIFY_FAILED aunque el navegador ande perfecto. Solo necesitás
   esto si te topás con ese error -- ver `tools/ca_bundle_windows.py`. El archivo está
   en el gitignore y es específico de cada máquina; su ausencia es el caso normal.

2b. Apunta también la caché de encodings de `tiktoken` dentro del repositorio (`.tiktoken_cache/`),
   por la misma razón: su valor por defecto es el directorio temporal del sistema operativo.

3. Apaga el paralelismo de `tokenizers`, que imprime una advertencia ruidosa en cada
   fork dentro de un notebook.

4. Detecta CUDA (Colab T4 es el entorno objetivo para la corrida a escala real) pero
   NO fija los hilos de BLAS en 1 cuando no está — una corrida SMOKE_TEST solo en CPU
   quiere todos los núcleos que pueda conseguir.

5. Fuerza el backend `Agg` (no interactivo) de matplotlib, pero SOLO fuera de un
   kernel de Jupyter/Colab. MEDIDO: ejecutar el `.py` de una etapa directamente como un script
   común (que es lo que hace una validación SMOKE_TEST en CPU) hereda el backend
   interactivo por defecto (TkAgg en una instalación típica de Windows), y `plt.show()`
   entonces se bloquea para siempre esperando una ventana que nunca nadie va a cerrar.
   Forzar `Agg` incondicionalmente rompería en cambio el ploteo inline real de Colab,
   así que esto solo entra en acción cuando `get_ipython()` no existe.

6. Reconfigura stdout a UTF-8. La consola de Windows es cp1252 por defecto, y
   los tokens de BPE a nivel byte (de la Etapa 1 en adelante) imprimen caracteres como 'Ġ' (U+0120) o,
   después de decodificar una generación todavía incoherente, el carácter de reemplazo
   '�' (U+FFFD) -- ambos rompen un `print()` común en esa plataforma si no.

7. Crea `checkpoints/`, el directorio a través del cual cada etapa se pasa artefactos.
   Está en el gitignore, así que NO existe en un clon nuevo, y el primer
   `plt.savefig("checkpoints/...")` levantaría un FileNotFoundError si no.

Todo es idempotente: una variable ya definida desde afuera se deja como está.

IMPORTÁ ESTO PRIMERO. El punto 1 solo funciona si corre antes de que se importen `datasets` /
`huggingface_hub` -- ellos leen HF_HOME hacia constantes a nivel de módulo
en su propio momento de import, así que definirla después no hace nada, en silencio, y
tus descargas terminan en la caché global. `_warn_if_hf_already_imported`
más abajo convierte esa falla silenciosa en una advertencia visible.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HF_CACHE = ROOT / ".hf_cache"
TIKTOKEN_CACHE = ROOT / ".tiktoken_cache"
CA_BUNDLE = ROOT / ".certs" / "ca_bundle.pem"
CHECKPOINTS = ROOT / "checkpoints"


def _set_if_missing(key: str, value: str) -> bool:
    """Define una variable de entorno solo si no estaba ya definida. Devuelve True si la definió."""
    if os.environ.get(key):
        return False
    os.environ[key] = value
    return True


def _warn_if_hf_already_imported() -> None:
    """
    HF_HOME se lee una sola vez, en el momento del import de `huggingface_hub`. Si algo ya
    lo importó, definir la variable ahora no hace nada y la caché local del repositorio queda
    silenciosamente ignorada -- una falla sin más síntoma que un perfil de usuario lleno.
    """
    already = sorted(m for m in ("huggingface_hub", "datasets") if m in sys.modules)
    if already and not os.environ.get("HF_HOME"):
        was = "fue" if len(already) == 1 else "fueron"
        print(
            f"ADVERTENCIA: {', '.join(already)} {was} importado antes que `environment`, así que la "
            f"caché local de HuggingFace del repositorio ({HF_CACHE}) NO está en efecto y las descargas "
            f"van a ir a la caché global por defecto. Importá `environment` primero.",
            file=sys.stderr,
        )


# --- 1. caché de HuggingFace dentro del repositorio -------------------------
_warn_if_hf_already_imported()
HF_CACHE.mkdir(parents=True, exist_ok=True)
_set_if_missing("HF_HOME", str(HF_CACHE))

# --- 1b. el directorio a través del cual cada etapa se pasa artefactos ------
# Está en el gitignore, así que un clon nuevo no lo tiene. La Etapa 0 escribe un PNG acá antes
# de que nada más lo cree.
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

# --- 2. certificados del proxy corporativo ----------------------------------
CA_BUNDLE_ACTIVE = False
if CA_BUNDLE.is_file():
    _set_if_missing("SSL_CERT_FILE", str(CA_BUNDLE))
    _set_if_missing("REQUESTS_CA_BUNDLE", str(CA_BUNDLE))
    _set_if_missing("CURL_CA_BUNDLE", str(CA_BUNDLE))
    CA_BUNDLE_ACTIVE = True

# --- 2b. la caché de tiktoken, también dentro del repositorio ---------------
# `tiktoken.get_encoding` descarga su archivo de encoding una vez. Si se lo deja solo, cae
# en el directorio temporal del sistema operativo, fuera del repositorio, y desaparece con la limpieza de temporales, así que
# la llamada de red se repite. El mismo razonamiento que el punto 1.
TIKTOKEN_CACHE.mkdir(parents=True, exist_ok=True)
_set_if_missing("TIKTOKEN_CACHE_DIR", str(TIKTOKEN_CACHE))

# --- 3. ruido de la advertencia de fork de tokenizers -----------------------
_set_if_missing("TOKENIZERS_PARALLELISM", "false")

# --- 4. datasets: sin barras de progreso duplicadas en un notebook ----------
_set_if_missing("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# --- 5. backend no interactivo de matplotlib, solo fuera de un kernel de notebook -
try:
    get_ipython()  # type: ignore[name-defined]  # definida solo dentro de IPython/Jupyter/Colab
except NameError:
    import matplotlib

    matplotlib.use("Agg")

# --- 6. stdout en UTF-8, para que los tokens de BPE a nivel byte impriman limpio en Windows -----
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def has_cuda() -> bool:
    """True si hay una GPU NVIDIA usable. Import diferido: no cargar torch sin necesidad."""
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def device() -> str:
    """'cuda' o 'cpu'."""
    return "cuda" if has_cuda() else "cpu"


def smoke_test(default: bool) -> bool:
    """
    `SMOKE_TEST` de una etapa. La variable de entorno `LAB_SMOKE_TEST` (1/0) manda sobre `default`: así
    `run_all.py` corre todas las etapas en el mismo modo. Sin la variable, cada etapa decide como siempre.
    """
    value = os.environ.get("LAB_SMOKE_TEST", "").strip().lower()
    if value in {"1", "true", "si", "sí", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    return default


def diagnostics() -> dict:
    """Reemplazo en Python de `!nvidia-smi`. Funciona bien sin GPU también."""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "venv": sys.prefix,
        "cpu": platform.processor() or "desconocido",
        "logical_cores": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            info["gpu_memory_gb"] = round(total / 1e9, 2)
    except ImportError:
        info["torch"] = "no instalado"

    try:
        import psutil

        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["ram_free_gb"] = round(psutil.virtual_memory().available / 1e9, 1)
    except ImportError:
        pass

    return info


def process_ram_gb() -> float:
    """
    Memoria residente de ESTE proceso, en GB.

    En Colab con GPU, el equivalente es `torch.cuda.memory_allocated()`.
    Sin GPU, el análogo honesto es cuánta RAM tiene tomada el proceso de
    Python mismo.
    """
    try:
        import psutil
    except ImportError:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def summary() -> None:
    """Imprime los diagnósticos, de forma legible."""
    info = diagnostics()
    width = max(len(k) for k in info)
    print("=" * 62)
    print("ENTORNO")
    print("=" * 62)
    for k, v in info.items():
        print(f"{k:<{width}} : {v}")
    print("-" * 62)
    print(f"{'HF_HOME':<{width}} : {os.environ.get('HF_HOME')}")
    print(
        f"{'ca_bundle':<{width}} : "
        + (f"activo ({CA_BUNDLE})" if CA_BUNDLE_ACTIVE else "no presente (normal)")
    )
    if not info.get("cuda_available", False):
        print()
        print("No se detectó GPU NVIDIA: usá SMOKE_TEST = True y corré en CPU.")
        print("No es un error -- pero el perfil a escala completa necesita un entorno con GPU.")
    print("=" * 62)


if __name__ == "__main__":
    summary()
