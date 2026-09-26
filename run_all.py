"""
Corre todo el pipeline en tu máquina, en orden, y deja los notebooks ejecutados (con salidas) listos
para entregar.

    python run_all.py                    # modo automático: completo si hay GPU NVIDIA, prueba si no
    python run_all.py --modo prueba      # SMOKE_TEST en todas las etapas: valida en minutos
    python run_all.py --modo completo    # la corrida larga (en CPU tarda muchísimo; pide confirmación)
    python run_all.py --desde 4          # retomar desde una etapa (p. ej. después de un error)
    python run_all.py --solo 5           # una sola etapa
    python run_all.py --sin-juez         # saltear la Etapa 5 (si no tenés Ollama)

QUÉ HACE
--------
1. Chequea Python, dependencias, GPU, acceso a HuggingFace y (si va a correr la Etapa 5) Ollama.
2. Ejecuta `00_dataset.ipynb` … `05_judge.ipynb` con un kernel de Jupyter, mostrando en la consola lo
   que va imprimiendo cada celda. Todas las etapas corren en el mismo modo (variable `LAB_SMOKE_TEST`).
3. Guarda cada notebook ejecutado en `outputs/notebooks/` (también si falla, hasta la celda del error).
4. Al final arma `outputs/resultados.zip` con los notebooks ejecutados, las tablas y los gráficos
   (sin los `.pt`, que no van en la entrega).

Las etapas largas retoman solas: la Etapa 2 sigue desde su último checkpoint, la 4 carga los modelos si
ya están entrenados y el juez de la 5 no repite llamadas. Si algo se corta, `--desde N` y listo.

La Etapa 1 se saltea si ya existe `checkpoints/tokenizer.json`: volver a entrenar el tokenizer invalida
todos los modelos entrenados con el anterior. Para forzarlo, `--reentrenar-tokenizer`.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
NB_OUT_DIR = OUT_DIR / "notebooks"
CHECKPOINTS = ROOT / "checkpoints"

STAGES = {
    0: ("00_dataset", "Datos (inspección)"),
    1: ("01_tokenizer", "Tokenizador BPE"),
    2: ("02_pretraining", "Preentrenamiento + ablación ancho/profundo"),
    3: ("03_embeddings", "Análisis de embeddings"),
    4: ("04_sft", "SFT completo + LoRA"),
    5: ("05_judge", "Juez local (Ollama + qwen3:4b)"),
}
REQUIRED_MODULES = ["torch", "tokenizers", "datasets", "huggingface_hub", "matplotlib", "pandas", "numpy",
                    "sklearn", "psutil", "ipykernel"]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # consola de Windows (cp1252)
except (AttributeError, OSError):
    pass


def say(msg: str = "") -> None:
    print(msg, flush=True)


def banner(msg: str) -> None:
    say("\n" + "=" * 78)
    say(msg)
    say("=" * 78)


# --- chequeos previos ---------------------------------------------------------------------------

def check_python() -> None:
    if sys.version_info < (3, 10):
        sys.exit(f"Hace falta Python 3.10 o más nuevo (tenés {sys.version.split()[0]}).")


def check_modules() -> None:
    missing = [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        sys.exit(
            f"Faltan paquetes: {', '.join(missing)}.\n"
            f"Instalalos con:  {sys.executable} -m pip install -r requirements.txt"
        )
    if importlib.util.find_spec("nbclient") is None:
        say("Falta nbclient (lo que ejecuta los notebooks): se instala ahora.")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nbclient"], check=True)


def has_cuda() -> bool:
    """
    Se chequea en un proceso aparte y con límite de tiempo: con algunos drivers (sobre todo laptops con
    GPU híbrida) inicializar CUDA puede colgarse, o un proceso de Python anterior quedó trabado y tiene la
    GPU tomada. Así el script avisa en lugar de quedarse mudo para siempre.
    """
    say("Chequeando la GPU (hasta 90 s)...")
    code = "import torch; print('CUDA', torch.cuda.is_available(), flush=True)"
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
        out = r.stdout
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        if "CUDA True" in out:
            say("AVISO: CUDA anda, pero el proceso de prueba no terminó al cerrarse (problema del driver al salir).")
            say("       Se sigue igual; si una etapa no termina, avisá.")
            return True
        sys.exit(
            "La GPU no respondió en 90 s. Lo más común: quedó un proceso de Python colgado usándola.\n"
            "  1. Mirá qué la está usando:   nvidia-smi   y   ps aux | grep python\n"
            "  2. Matá los procesos colgados: kill -9 <PID>   (o reiniciá la máquina)\n"
            "  3. Volvé a correr este script."
        )
    if "CUDA" not in out:
        sys.exit(f"No se pudo importar torch:\n{r.stderr[-2000:]}")
    return "CUDA True" in out


def check_huggingface() -> None:
    try:
        from huggingface_hub import HfApi

        HfApi().dataset_info("roneneldan/TinyStories", timeout=15)
    except Exception as e:  # noqa: BLE001
        sys.exit(
            f"No se pudo llegar a HuggingFace ({type(e).__name__}: {e}).\n"
            "Revisá la conexión. Si estás en una red que inspecciona TLS (error CERTIFICATE_VERIFY_FAILED), "
            "mirá tools/ca_bundle_windows.py."
        )


def ollama_available() -> bool:
    return shutil.which("ollama") is not None


def pick_kernel() -> str:
    """El kernel tiene que usar ESTE Python (el del venv). Si 'python3' apunta a otro, se registra uno
    propio dentro del venv (--sys-prefix: no toca nada fuera de él)."""
    from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel

    try:
        argv0 = KernelSpecManager().get_kernel_spec("python3").argv[0]
    except NoSuchKernel:
        argv0 = ""
    if argv0 in ("python", "python3", sys.executable) or (argv0 and Path(argv0).resolve() == Path(sys.executable).resolve()):
        return "python3"
    name = "lab-proyecto"
    subprocess.run(
        [sys.executable, "-m", "ipykernel", "install", "--sys-prefix", "--name", name,
         "--display-name", "Python (proyecto lab)"],
        check=True, stdout=subprocess.DEVNULL,
    )
    return name


# --- ejecución de un notebook ----------------------------------------------------------------------

def run_notebook(stem: str, kernel_name: str, log) -> bool:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    src = ROOT / f"{stem}.ipynb"
    dst = NB_OUT_DIR / f"{stem}.ipynb"
    nb = nbformat.read(src, as_version=4)
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")

    class StreamingClient(NotebookClient):
        """Muestra en la consola lo que imprime cada celda, a medida que lo imprime."""

        def process_message(self, msg, cell, cell_index):
            result = super().process_message(msg, cell, cell_index)
            kind, content = msg.get("msg_type"), msg.get("content", {})
            if kind == "stream":
                text = content.get("text", "")
                sys.stdout.write(text)
                sys.stdout.flush()
                log.write(text)
            elif kind == "error":
                text = f"\n{content.get('ename')}: {content.get('evalue')}\n"
                sys.stdout.write(text)
                log.write(text)
            return result

    counter = {"n": 0}

    def on_cell_start(cell, cell_index, **_):
        if cell.cell_type == "code":
            counter["n"] += 1
            line = f"\n--- [{stem}] celda {counter['n']}/{n_code} ---\n"
            sys.stdout.write(line)
            log.write(line)

    client = StreamingClient(
        nb,
        kernel_name=kernel_name,
        timeout=None,  # hay celdas de entrenamiento de horas
        startup_timeout=180,
        resources={"metadata": {"path": str(ROOT)}},
        on_cell_start=on_cell_start,
    )
    ok = True
    try:
        client.execute()
    except CellExecutionError:
        ok = False
    except KeyboardInterrupt:
        ok = False
        say("\nInterrumpido a mano.")
    finally:
        nbformat.write(nb, dst)  # con las salidas hasta donde llegó
        log.flush()
    return ok


def bundle_results(smoke: bool) -> Path:
    """Notebooks ejecutados + tablas/gráficos/JSON de checkpoints/ (del modo que corrió), sin los .pt."""
    zip_path = OUT_DIR / ("resultados_prueba.zip" if smoke else "resultados.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for nb in sorted(NB_OUT_DIR.glob("*.ipynb")):
            z.write(nb, f"notebooks/{nb.name}")
        for f in sorted(CHECKPOINTS.rglob("*")):
            rel = f.relative_to(CHECKPOINTS)
            in_smoke = rel.parts[0] == "smoke"
            # Prueba: lo de checkpoints/smoke/ + lo que las Etapas 0 y 1 dejan arriba. Completo: todo menos smoke/.
            wanted = (in_smoke or len(rel.parts) == 1) if smoke else not in_smoke
            if wanted and f.is_file() and f.suffix.lower() in {".csv", ".png", ".json", ".jsonl"} and f.name != "tokenizer.json":
                z.write(f, f"checkpoints/{rel.as_posix()}")
        log = OUT_DIR / "run_all.log"
        if log.is_file():
            z.write(log, "run_all.log")
    return zip_path


# --- main -------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Corre el pipeline completo (Etapas 0-5) y guarda los notebooks ejecutados.")
    ap.add_argument("--modo", choices=["auto", "prueba", "completo"], default="auto",
                    help="auto: completo con GPU NVIDIA, prueba sin GPU")
    ap.add_argument("--desde", type=int, default=0, choices=sorted(STAGES), help="primera etapa a correr")
    ap.add_argument("--hasta", type=int, default=5, choices=sorted(STAGES), help="última etapa a correr")
    ap.add_argument("--solo", type=int, choices=sorted(STAGES), help="correr una sola etapa")
    ap.add_argument("--sin-juez", action="store_true", help="saltear la Etapa 5")
    ap.add_argument("--reentrenar-tokenizer", action="store_true",
                    help="correr la Etapa 1 aunque ya exista tokenizer.json (invalida los modelos entrenados)")
    ap.add_argument("--si", action="store_true", help="no pedir confirmación")
    ap.add_argument("--sin-chequeo-red", action="store_true",
                    help="no chequear HuggingFace antes de arrancar (si ya tenés los datos en caché)")
    args = ap.parse_args()

    os.chdir(ROOT)  # las etapas usan rutas relativas a la raíz del repositorio
    say("Preparando la corrida...")
    check_python()
    check_modules()
    cuda = has_cuda()
    smoke = {"auto": not cuda, "prueba": True, "completo": False}[args.modo]

    stages = [args.solo] if args.solo is not None else list(range(args.desde, args.hasta + 1))
    if args.sin_juez and 5 in stages:
        stages.remove(5)
    tokenizer = CHECKPOINTS / "tokenizer.json"
    if 1 in stages and tokenizer.is_file() and not args.reentrenar_tokenizer:
        stages.remove(1)
        say(f"Etapa 1 salteada: ya existe {tokenizer.relative_to(ROOT)} (usá --reentrenar-tokenizer para rehacerlo).")
    if 1 not in stages and any(s >= 2 for s in stages) and not tokenizer.is_file():
        stages.insert(0, 1)
        say("No hay tokenizer.json todavía: se agrega la Etapa 1.")

    banner("Corrida del pipeline")
    say(f"Python      : {sys.version.split()[0]} ({sys.executable})")
    say(f"GPU NVIDIA  : {'sí' if cuda else 'no'}")
    say(f"Modo        : {'PRUEBA (SMOKE_TEST: números sin valor, solo valida que corre)' if smoke else 'COMPLETO (la corrida larga)'}")
    say(f"Etapas      : {', '.join(f'{s} ({STAGES[s][1]})' for s in stages)}")

    if not smoke and not cuda:
        say("\nATENCIÓN: modo completo sin GPU NVIDIA. La Etapa 2 sola puede tardar un día entero en CPU.")
    if not smoke and 5 in stages and not ollama_available():
        say("\nLa Etapa 5 necesita Ollama y no está instalado.")
        say("  Instalalo desde https://ollama.com/download y volvé a correr, o usá --sin-juez.")
        return 1
    if not smoke and not args.si:
        if input("\n¿Arrancar? [s/N] ").strip().lower() not in {"s", "si", "sí", "y", "yes"}:
            return 0

    if not args.sin_chequeo_red:
        check_huggingface()
    kernel = pick_kernel()
    os.environ["LAB_SMOKE_TEST"] = "1" if smoke else "0"  # lo heredan los kernels: mismo modo en todas las etapas
    NB_OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    with open(OUT_DIR / "run_all.log", "a", encoding="utf-8") as log:
        log.write(f"\n\n##### corrida {time.strftime('%Y-%m-%d %H:%M:%S')} · modo {'prueba' if smoke else 'completo'} · etapas {stages}\n")
        for s in stages:
            stem, title = STAGES[s]
            banner(f"Etapa {s} · {title}  ({stem}.ipynb)")
            t0 = time.perf_counter()
            ok = run_notebook(stem, kernel, log)
            minutes = (time.perf_counter() - t0) / 60
            if not ok:
                banner(f"FALLÓ la Etapa {s} después de {minutes:.1f} min")
                say(f"El notebook con las salidas hasta el error quedó en outputs/notebooks/{stem}.ipynb")
                say(f"Arreglá el problema y retomá con:  python run_all.py --modo {'prueba' if smoke else 'completo'} --desde {s}")
                return 1
            say(f"\nEtapa {s} lista en {minutes:.1f} min → outputs/notebooks/{stem}.ipynb")

    zip_path = bundle_results(smoke)
    banner(f"Todo listo en {(time.perf_counter() - t_start) / 60:.1f} min")
    say(f"Notebooks ejecutados : {NB_OUT_DIR.relative_to(ROOT)}/")
    say(f"Resultados para el informe (sin .pt): {zip_path.relative_to(ROOT)}")
    if smoke:
        say("Fue una corrida de PRUEBA: los números no significan nada. Para la de verdad: --modo completo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
