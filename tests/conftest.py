"""
Local test bootstrap.

At runtime the Docker image does `COPY . ./pineforge_ai`, so the repo root IS
the `pineforge_ai` package (PYTHONPATH=/app). Modules import each other as
`pineforge_ai.<module>`. Locally the repo dir is named `AI_trader`, so we alias
`pineforge_ai` onto the repo root here — letting tests import `pineforge_ai.*`
exactly like production while the bare `indicators.*` imports keep working too.
"""
import pathlib
import sys
import types

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if "pineforge_ai" not in sys.modules:
    _pkg = types.ModuleType("pineforge_ai")
    _pkg.__path__ = [str(_ROOT)]
    sys.modules["pineforge_ai"] = _pkg
