"""Vendors jieba's dictionary and HMM model into lilia/data/jieba.

Usage: python scripts/jieba_data.py <path to an unpacked jieba source tree>

dict.txt is copied as it is. The HMM probabilities (finalseg/prob_*.py,
Python dict literals) become one entry per line, floats written with repr()
so the C++ side reads back the same doubles:

    start <state> <logprob>
    trans <from> <to> <logprob>
    emit <state> <character> <logprob>
"""

import ast
import shutil
import sys
from pathlib import Path


def load(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "P":
            return ast.literal_eval(node.value)
    raise SystemExit(f"{path}: no P = {{...}}")


def main() -> None:
    source = Path(sys.argv[1]) / "jieba"
    out = Path(__file__).resolve().parent.parent / "lilia" / "data" / "jieba"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "dict.txt", out / "jieba_dict.txt")
    finalseg = source / "finalseg"
    lines = []
    for state, p in load(finalseg / "prob_start.py").items():
        lines.append(f"start {state} {p!r}")
    for state, row in load(finalseg / "prob_trans.py").items():
        for to, p in row.items():
            lines.append(f"trans {state} {to} {p!r}")
    for state, row in load(finalseg / "prob_emit.py").items():
        for char, p in row.items():
            if char.isspace() or len(char) != 1:
                raise SystemExit(f"unexpected emission key {char!r}")
            lines.append(f"emit {state} {char} {p!r}")
    (out / "jieba_hmm.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
