from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent / "kernels"

for gen in sorted(root.glob("*/gen.py")):
    print(f"[gen] {gen}")
    subprocess.check_call([sys.executable, str(gen)])