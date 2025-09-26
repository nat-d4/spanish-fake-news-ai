#shared helpers for IO, cleaning, and simple logging
import os 
import re
import pandas as pd

ANSI = {
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "red": "\x1b[31m",
    "cyan": "\x1b[36m",
    "reset": "\x1b[0m"
}

def log(msg: str, level: str = "info") -> None:
    color = {
        "info": ANSI["cyan"],
        "ok": ANSI["green"],
        "warn": ANSI["yellow"],
        "err": ANSI["red"],
    }.get(level, "")
    print(f"{color}[{level.upper()}]{ANSI['reset']} {msg}")

_slug_re = re.compile(r"[a-z0-9]+")

def slugify(text: str, max_len: int = 60) -> None:
    s = text.lower()
    s = _slug_re.sub("-", s).strip("-")
    return s[:max_len]

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok = True)

def write_csv_safe(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index = False)