import sys
import os


# This adds the project root directory (the parent of 'src') to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

RED = "\x1b[1;31m"
BLUE = "\x1b[1;34m"
GREEN = "\x1b[1;32m"
RESET = "\x1b[0m"
INFO = f"{BLUE}[INFO]{RESET}"
OK = f"{GREEN}[OK]{RESET}"
KO = f"{RED}[KO]{RESET}"