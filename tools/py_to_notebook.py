"""
Convierte un `.py` con marcadores de celda `# %%` en un `.ipynb` de verdad.

    python tools/py_to_notebook.py                  # regenerar todas las etapas, conservando salidas
    python tools/py_to_notebook.py --check           # fallar si algún .ipynb quedó desactualizado
    python tools/py_to_notebook.py --only 02_pretraining
    python tools/py_to_notebook.py --limpiar         # regenerar SIN salidas (notebook sin ejecutar)
    python tools/py_to_notebook.py --bootstrap-only   # actualizar solo el arranque; conservar salidas
    python tools/py_to_notebook.py --source foo.py --destination foo.ipynb

POR QUÉ EXISTE ESTE PASO
--------------------
El `.py` es la fuente EDITABLE: diffea limpio en git, se puede grepear normalmente y
no arrastra metadata ni salidas viejas. El `.ipynb` es el ARTEFACTO que abrís
y ejecutás. Editar el `.py` y regenerar es muchísimo más sano que editar
JSON a mano.

SALIDAS DE UN NOTEBOOK YA EJECUTADO
-----------------------------------
Los `.ipynb` de la entrega van ejecutados, con sus salidas. Regenerar conserva, celda por celda,
las salidas de cada celda de código cuyo código NO cambió (y los ids y la metadata del notebook):
así se puede corregir o ampliar el markdown del `.py` sin volver a correr una etapa de horas. Una
celda de código que sí cambió queda sin salidas, y el script lo avisa: esa etapa hay que volver a
ejecutarla. `--limpiar` descarta todas las salidas. `--check` compara solo el tipo y el texto de
cada celda, así que un notebook ejecutado que está al día con su `.py` pasa el chequeo.

(VS Code también puede abrir el `.py` directamente como notebook interactivo, así que si
con eso te alcanza, este script es opcional.)

FORMATO ESPERADO
---------------
    # %%                 -> celda de código
    # %% [markdown]      -> celda de markdown, cuyo contenido es UN único string entre
                            triples comillas (se acepta el prefijo `r`), para que la prosa
                            no tenga un `#` delante de cada línea.

Todo lo que está ANTES del primer `# %%` se descarta: es el encabezado del archivo
fuente, no una celda.

ARRANQUE EN COLAB
---------------
Todo `.ipynb` generado recibe una primera celda extra (`BOOTSTRAP_CELL`, más abajo) que
clona el repositorio, hace chdir hacia él e instala `requirements.txt` con pip -- pero solo
cuando detecta Colab. Se inyecta acá en lugar de escribirse dentro del `.py` de cada
etapa para que la URL de clonado viva en un único lugar y los seis notebooks no puedan
divergir. **Definí `REPO_URL` más abajo antes de publicar.**

Solo librería estándar: no depende de jupytext ni de nbformat.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# stem -> (fuente .py, destino .ipynb), un par por etapa del pipeline.
STAGES: dict[str, tuple[Path, Path]] = {
    stem: (ROOT / f"{stem}.py", ROOT / f"{stem}.ipynb")
    for stem in (
        "00_dataset",
        "01_tokenizer",
        "02_pretraining",
        "03_embeddings",
        "04_sft",
        "05_judge",
    )
}

CELL_MARKER = re.compile(r"^#\s*%%(.*)$")

# --- arranque en Colab ------------------------------------------------------
# EDITÁ ESTAS DOS LÍNEAS cuando publiques el repositorio. Son el ÚNICO lugar donde aparece
# la URL de clonado: la celda de arranque de más abajo se inyecta como celda 1 de cada
# .ipynb generado, así que los seis notebooks quedan sincronizados automáticamente.
URL_PLACEHOLDER = "CHANGEME"
REPO_URL = "https://github.com/facurie/Proyecto1_IA_Generativa.git"
REPO_DIR = "Proyecto1_IA_Generativa"
# El repositorio tiene que ser PÚBLICO: el `git clone` del arranque es sin autenticación, así que una
# URL privada falla en Colab exactamente igual que una equivocada.

# Se antepone a cada notebook. Es Python deliberadamente liso (sin magics `!`), para que
# sea válido tanto en Colab como en Jupyter y en VS Code, y no haga nada fuera de Colab,
# donde el estudiante ya tiene el repositorio clonado y las dependencias instaladas.
BOOTSTRAP_CELL = f'''# --- arranque en Colab (no hace nada fuera de Colab) -----------------------
# Colab arranca en /content con un entorno vacío: sin esta celda el
# `import environment` siguiente falla con ModuleNotFoundError.
import os
import subprocess
import sys

if "google.colab" in sys.modules:
    REPO_URL = "{REPO_URL}"
    REPO_DIR = "{REPO_DIR}"

    repo_path = os.path.join("/content", REPO_DIR)
    if not os.path.isdir(repo_path):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, repo_path], check=True)
    os.chdir(repo_path)
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True
    )

    # El disco de Colab es EFÍMERO: cada checkpoint se pierde cuando el entorno de ejecución
    # se desconecta (~90 min de inactividad en el plan gratuito), y las Etapas 3, 4 y 5 necesitan
    # todas los checkpoints de la Etapa 2. Descomentá para guardarlos en Drive en su lugar.
    #
    # Notar `islink`, no `exists`: importar `environment` crea un directorio checkpoints/
    # REAL, y `exists` entonces saltearía el enlace en silencio y mandaría cada
    # checkpoint de vuelta al disco efímero.
    #
    # from google.colab import drive
    # drive.mount("/content/drive")
    # PERSISTENT = "/content/drive/MyDrive/Proyecto1_IA_Generativa/checkpoints"
    # os.makedirs(PERSISTENT, exist_ok=True)
    # if os.path.islink("checkpoints"):
    #     print("checkpoints ->", os.readlink("checkpoints"))
    # elif os.path.isdir("checkpoints"):
    #     print("ADVERTENCIA: checkpoints/ ya es un directorio real, así que NO está en Drive.")
    #     print("Mové su contenido a", PERSISTENT, ", borralo, y volvé a ejecutar esta celda.")
    # else:
    #     os.symlink(PERSISTENT, "checkpoints")
    #     print("checkpoints ->", PERSISTENT)

    print("Arranque en Colab OK. Directorio de trabajo:", os.getcwd())
'''


def _to_lines(text: str) -> list[str]:
    """El ipynb guarda `source` como una lista de líneas, cada una terminada en \\n."""
    if not text:
        return []
    lines = text.splitlines()
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def _trim(block: list[str]) -> str:
    """Elimina las líneas en blanco del principio y del final."""
    while block and not block[0].strip():
        block.pop(0)
    while block and not block[-1].strip():
        block.pop()
    return "\n".join(block)


def split_into_cells(source: str) -> list[tuple[str, str]]:
    """Devuelve [(tipo, contenido), ...] con tipo en {'code', 'markdown'}."""
    cells: list[tuple[str, str]] = []
    current_kind: str | None = None
    buffer: list[str] = []

    def close():
        if current_kind is None:
            return
        content = _trim(list(buffer))
        if not content:
            return
        if current_kind == "markdown":
            content = _text_from_string_literal(content)
        cells.append((current_kind, content))

    for line in source.splitlines():
        m = CELL_MARKER.match(line)
        if m:
            close()
            current_kind = "markdown" if "[markdown]" in m.group(1) else "code"
            buffer = []
        elif current_kind is not None:
            buffer.append(line)

    close()
    return cells


def _text_from_string_literal(content: str) -> str:
    """
    El cuerpo de una celda de markdown es un string literal de Python. Lo parseamos con `ast` para
    quedarnos con el texto mismo, sin el prefijo `r` pegado adelante.
    """
    try:
        node = ast.parse(content).body
    except SyntaxError as e:
        raise SystemExit(
            f"celda de markdown mal formada (no parsea como Python): {e}\n"
            f"empieza con: {content[:80]!r}"
        ) from e

    if len(node) == 1 and isinstance(node[0], ast.Expr) and isinstance(node[0].value, ast.Constant):
        value = node[0].value.value
        if isinstance(value, str):
            return value.strip("\n")

    raise SystemExit(
        "una celda `# %% [markdown]` tiene que contener exactamente UN string entre "
        'triples comillas (recomendado: r"""..."""). '
        f"Esta empieza con: {content[:80]!r}"
    )


def build_notebook(cells: list[tuple[str, str]]) -> dict:
    output = []
    for i, (kind, content) in enumerate(cells, start=1):
        cell = {
            "id": f"cell-{i:03d}",
            "cell_type": kind,
            "metadata": {},
            "source": _to_lines(content),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        output.append(cell)

    return {
        "cells": output,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _cell_sources(nb: dict) -> list[tuple[str, str]]:
    """Lo que define el contenido de un notebook para `--check`: tipo y texto de cada celda."""
    return [(cell["cell_type"], "".join(cell["source"])) for cell in nb["cells"]]


def _dump(nb: dict) -> str:
    """Mismo formato que `nbformat.write` (claves ordenadas), para que git diffee solo lo que cambió."""
    return json.dumps(nb, sort_keys=True, ensure_ascii=False, indent=1) + "\n"


def carry_over(nb: dict, previous: dict) -> tuple[int, str | None, int]:
    """
    Copia a `nb`, recién generado desde el `.py`, lo que ya tenía la versión ejecutada del notebook:
    las salidas y el número de ejecución de cada celda de código cuyo código no cambió, el id de las
    celdas que siguen iguales y la metadata del notebook (kernel, versión de Python).

    Las salidas se copian solo mientras las celdas de código coinciden, en orden, con las del
    notebook ejecutado. Desde la primera celda de código nueva, cambiada, borrada o movida, ninguna
    conserva salidas: las celdas siguientes corren sobre el estado que deja esa, así que sus salidas
    viejas podrían no corresponder al código nuevo.

    Devuelve cuántas celdas conservaron salidas, el comienzo de la primera celda de código que dejó
    de coincidir (o None) y cuántas celdas de código quedaron sin salidas desde ahí.
    """
    source = lambda cell: "".join(cell["source"])  # noqa: E731
    old_code = [cell for cell in previous.get("cells", []) if cell["cell_type"] == "code"]
    by_content: dict[tuple[str, str], list[dict]] = {}
    for cell in previous.get("cells", []):
        by_content.setdefault((cell["cell_type"], source(cell)), []).append(cell)

    # Ids: se conserva el de cada celda que sigue igual; las nuevas reciben uno estable, derivado del
    # contenido, así regenerar dos veces el mismo `.py` no cambia nada.
    reused = [(by_content.get((cell["cell_type"], source(cell))) or [None]).pop(0) for cell in nb["cells"]]
    old_ids = {old["id"] for old in reused if old is not None and old.get("id")}
    taken: set[str] = set()
    for cell, old in zip(nb["cells"], reused):
        if old is not None and old.get("id") and old["id"] not in taken:
            cell["id"] = old["id"]
        else:
            digest = hashlib.sha1(f"{cell['cell_type']}|{source(cell)}".encode()).hexdigest()[:8]
            cell["id"], n = f"cell-{digest}", 2
            while cell["id"] in taken or cell["id"] in old_ids:
                cell["id"], n = f"cell-{digest}-{n}", n + 1
        taken.add(cell["id"])

    code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    same = 0
    while same < min(len(code_cells), len(old_code)) and source(code_cells[same]) == source(old_code[same]):
        same += 1
    kept = 0
    for cell, old in zip(code_cells[:same], old_code):
        cell["execution_count"] = old.get("execution_count")
        cell["outputs"] = old.get("outputs", [])
        cell["metadata"] = old.get("metadata", {})
        kept += bool(cell["outputs"])
    first_changed = None
    if same < len(code_cells):
        first_changed = source(code_cells[same]).splitlines()[0][:70] if code_cells[same]["source"] else "(vacía)"

    nb["metadata"] = previous.get("metadata") or nb["metadata"]
    return kept, first_changed, len(code_cells) - same


def update_bootstrap(destination: Path, check: bool) -> bool:
    """Actualiza solo la primera celda de un notebook ejecutado, sin tocar sus resultados."""
    if not destination.is_file():
        print(f"FALTA {destination.name}: ejecutá el script sin --bootstrap-only.")
        return False

    nb = json.loads(destination.read_text(encoding="utf-8"))
    first = nb["cells"][0]
    if first["cell_type"] != "code" or "arranque en Colab" not in "".join(first["source"][:1]):
        print(f"{destination.name}: la primera celda no es el arranque de Colab esperado.")
        return False

    source = _to_lines(BOOTSTRAP_CELL.rstrip("\n"))
    if check:
        if first["source"] != source:
            print(f"{destination.name}: arranque de Colab desactualizado.")
            return False
        print(f"OK: arranque de {destination.name} actualizado; salidas conservadas.")
        return True

    first["source"] = source
    first["execution_count"] = None
    first["outputs"] = []
    first["metadata"] = {}
    destination.write_text(_dump(nb), encoding="utf-8")
    print(f"arranque actualizado: {destination.name}; otras celdas y salidas conservadas.")
    return True


def convert_one(source: Path, destination: Path, check: bool, clean: bool = False) -> bool:
    """
    Devuelve True si salió bien (o si el --check coincide limpio), False si falló. Sin `clean`, si el
    destino ya existe, conserva las salidas de las celdas de código que no cambiaron (`carry_over`).
    """
    if not source.is_file():
        print(f"falta la fuente: {source}")
        return False

    text = source.read_text(encoding="utf-8")
    cells = split_into_cells(text)
    if not cells:
        print(f"no se encontraron celdas `# %%` en {source}")
        return False

    # El arranque pertenece al ARTEFACTO, no a la fuente: ejecutar el `.py` de una etapa
    # localmente ya implica tener el repositorio en disco y el venv activo. Se antepone
    # acá y no dentro de build_notebook() para que los conteos de abajo describan el
    # archivo que efectivamente se escribe.
    cells = [("code", BOOTSTRAP_CELL.rstrip("\n"))] + cells

    nb = build_notebook(cells)

    n_md = sum(1 for k, _ in cells if k == "markdown")
    n_code = len(cells) - n_md

    if check:
        if not destination.is_file():
            print(f"FALTA {destination.name}: ejecutá el script sin --check.")
            return False
        # Solo el texto de las celdas: las salidas de un notebook ejecutado no son un desfasaje.
        if _cell_sources(json.loads(destination.read_text(encoding="utf-8"))) != _cell_sources(nb):
            print(f"{destination.name} está DESACTUALIZADO respecto de {source.name}. Ejecutá sin --check.")
            return False
        if URL_PLACEHOLDER in REPO_URL:
            print(
                f"{destination.name}: REPO_URL sigue siendo el placeholder {URL_PLACEHOLDER}, así que su "
                f"celda de arranque de Colab no puede clonar nada."
            )
            return False
        print(f"OK: {destination.name} coincide con {source.name} ({len(cells)} celdas).")
        return True

    kept, first_changed, dropped, was_executed = 0, None, 0, False
    if destination.is_file() and not clean:
        previous = json.loads(destination.read_text(encoding="utf-8"))
        was_executed = any(cell.get("outputs") for cell in previous.get("cells", []))
        kept, first_changed, dropped = carry_over(nb, previous)

    destination.write_text(_dump(nb), encoding="utf-8")
    print(f"escrito: {destination}")
    print(f"  celdas: {len(cells)}  ({n_code} de código, {n_md} de markdown)")
    if kept:
        print(f"  salidas conservadas: {kept} celdas de código")
    if was_executed and first_changed is not None:
        print(f"  ATENCIÓN: el código cambió desde la celda que empieza con {first_changed!r}; desde ahí,"
              f" {dropped} celda(s) de código quedaron sin salidas. Volvé a ejecutar el notebook.")
    print(f"  tamaño: {destination.stat().st_size:,} bytes")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="no escribir; fallar si algún .ipynb no coincide con su .py")
    ap.add_argument("--bootstrap-only", action="store_true",
                    help="actualizar/comprobar solo la primera celda sin borrar resultados ejecutados")
    ap.add_argument("--limpiar", action="store_true",
                    help="regenerar sin salidas, aunque el .ipynb ya esté ejecutado")
    ap.add_argument("--only", type=str, default=None,
                    help="restringir a una sola etapa por su stem, p. ej. 02_pretraining")
    ap.add_argument("--source", type=Path, default=None,
                    help="convertir un único archivo arbitrario en lugar de la lista STAGES")
    ap.add_argument("--destination", type=Path, default=None)
    args = ap.parse_args()

    if args.source is not None:
        destination = args.destination or args.source.with_suffix(".ipynb")
        if args.bootstrap_only:
            ok = update_bootstrap(destination, args.check)
        else:
            ok = convert_one(args.source, destination, args.check, clean=args.limpiar)
        return 0 if ok else 1

    pairs = STAGES.items() if args.only is None else {args.only: STAGES[args.only]}.items()

    all_ok = True
    any_found = False
    for stem, (source, destination) in pairs:
        if not source.is_file():
            continue  # etapa todavía no escrita -- no todas las etapas existen en todo momento de la construcción
        any_found = True
        result = (update_bootstrap(destination, args.check) if args.bootstrap_only
                  else convert_one(source, destination, args.check, clean=args.limpiar))
        all_ok = result and all_ok

    if not any_found:
        print("todavía no se encontró ningún .py de etapa.")
        return 1

    if URL_PLACEHOLDER in REPO_URL and not args.check:
        print()
        print("!" * 72)
        print(f"REPO_URL sigue siendo el placeholder {URL_PLACEHOLDER}. Los notebooks quedaron escritos,")
        print("pero su celda de arranque de Colab va a fallar en el `git clone` para cada estudiante.")
        print(f"Definí REPO_URL en {Path(__file__).name} con la URL PÚBLICA del repositorio y volvé a ejecutar.")
        print("`--check` sale con código distinto de cero hasta que lo hagas, así puede frenar la publicación.")
        print("!" * 72)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
