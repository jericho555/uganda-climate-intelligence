"""Launch the Uganda Climate Intelligence dashboard.

Prepends the workspace-local vendored packages (``._webpkgs``) onto
``sys.path`` so Streamlit can be imported even when system site-packages is
not writable, then hands control to the Streamlit CLI running ``app.py``.

Usage::

    python dashboard/run_dashboard.py            # default port 8501
    python dashboard/run_dashboard.py --server.port 8600
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VEND = ROOT / "._webpkgs"
APP = HERE / "app.py"

if VEND.exists() and str(VEND) not in sys.path:
    sys.path.insert(0, str(VEND))

try:
    from streamlit.web import cli as stcli
except ImportError:
    sys.stderr.write(
        "ERROR: streamlit is not importable.\n"
        f"Expected vendored packages at: {VEND}\n"
        "Install them with:\n"
        f"  python -m pip install --target={VEND.name} "
        "--no-cache-dir streamlit\n"
    )
    raise SystemExit(1)

if __name__ == "__main__":
    # Sensible defaults so the launcher works regardless of any ambient
    # ~/.streamlit/config.toml (e.g. developmentMode=true, which otherwise
    # forbids --server.port). User-supplied flags come last and win.
    defaults = [
        "--global.developmentMode=false",
        "--browser.gatherUsageStats=false",
    ]
    args = ["streamlit", "run", str(APP)] + defaults + sys.argv[1:]
    sys.argv = args
    sys.exit(stcli.main())
