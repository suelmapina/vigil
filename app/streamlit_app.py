"""Entry point for Streamlit Community Cloud.

The demo lives in src/growthlab/demo.py. This file makes the package importable without
installing the full modelling stack, then runs the demo on every Streamlit rerun.
"""

import runpy
import sys
from pathlib import Path

src = Path(__file__).resolve().parents[1] / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

runpy.run_path(str(src / "vigil" / "demo.py"), run_name="__main__")
