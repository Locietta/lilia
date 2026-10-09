"""Vendors pinyin readings for the typo generator into lilia/data/pinyin.

Usage: python scripts/pinyin_data.py <unpacked pypinyin source> <unpacked jieba source>

MaiBot's typo generator (src/chat/utils/typo_generator.py) asks pypinyin for
readings (Style.TONE3, first reading). Two files capture what it would get:

    pinyin_chars.txt  "<char> <reading>" for U+4E00..U+9FFE, the characters
                      MaiBot maps from pinyin to homophones (characters with
                      no reading are left out);
    pinyin_words.txt  "<word> <reading> <reading>..." for jieba dictionary
                      words of two or more Han characters whose reading in
                      context differs from their characters' first readings
                      (pypinyin's phrase dictionary, e.g. 银行 -> yin2 hang2).

Character frequencies need no file: MaiBot's depends-data/char_frequency.json
equals what its code derives from jieba's dictionary (checked 2026-10-09, all
12,010 characters), and lilia derives them the same way at runtime.
"""

import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from pypinyin import Style, pinyin  # noqa: E402


def is_han(char: str) -> bool:
    return "一" <= char <= "鿿"


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "lilia" / "data" / "pinyin"
    out.mkdir(parents=True, exist_ok=True)
    first = {}
    lines = []
    for code in range(0x4E00, 0x9FFF):
        char = chr(code)
        reading = pinyin(char, style=Style.TONE3)[0][0]
        if reading != char:
            first[char] = reading
            lines.append(f"{char} {reading}")
    (out / "pinyin_chars.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )

    words = []
    dictionary = Path(sys.argv[2]) / "jieba" / "dict.txt"
    seen = set()
    for line in dictionary.read_text(encoding="utf-8").splitlines():
        word = line.split(" ")[0]
        if len(word) < 2 or word in seen or not all(is_han(c) for c in word):
            continue
        seen.add(word)
        readings = [r[0] for r in pinyin(word, style=Style.TONE3)]
        if len(readings) == len(word) and readings != [first.get(c, c) for c in word]:
            words.append(f"{word} {' '.join(readings)}")
    (out / "pinyin_words.txt").write_text(
        "\n".join(words) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{len(lines)} characters, {len(words)} words with readings in context")


if __name__ == "__main__":
    main()
