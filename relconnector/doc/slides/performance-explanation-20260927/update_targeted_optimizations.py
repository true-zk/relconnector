"""Compatibility entry: rebuild the integrated narrative instead of appending update slides."""
from pathlib import Path
import subprocess,sys
from build_integrated_deck import update
if __name__=='__main__':
    subprocess.run([sys.executable,str(Path(__file__).with_name('build_update.py'))],check=True)
