"""Rebuild the historical reference together with current paired v2 evidence."""
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[4]
runpy.run_path(str(ROOT/'doc/slides/performance-explanation-20260927/build_integrated_figures.py'),run_name='__main__')
