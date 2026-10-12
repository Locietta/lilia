"""Build an offline page for labelling whether chat lines were said to lilia.

    python scripts/addressee_label_page.py --group <id> [--count 120] [--data-dir ~/.local/share/lilia] --out addressee-label.html

Picks recent lines of one group, preferring those the hindsight judge already
labelled (so `lilia feedback --import-manual` can report how often it agreed),
and shows each with the lines around it. Clicking marks a line as said to her
or not; the page exports labels.json ({"labels": [{chat_id, message_id,
to_her}]}). The page holds private chat lines: keep it out of the tree (ops/
is ignored) and delete it when done.
"""

import argparse
import html
import json
import os
import sqlite3

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>lilia addressee labels</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 860px; margin: 0 auto; padding: 12px; }
#bar { position: sticky; top: 0; background: #fff; padding: 8px 0; border-bottom: 1px solid #ddd; z-index: 1; }
.item { border: 1px solid #ddd; border-radius: 6px; margin: 10px 0; padding: 8px; }
.item.her { border-color: #2a2; background: #eef8ee; }
.item.not { border-color: #a22; background: #f8eeee; }
.line { font-size: 13px; color: #666; white-space: pre-wrap; }
.line.target { color: #000; font-weight: 600; background: #fff7cc; }
.line.self { color: #15c; }
button { margin-right: 6px; }
</style></head><body>
<div id="bar"><b>这句话是不是在对她（莉莉娅）说？</b> 黄底那行是要判断的；看前后文判断。
  <span id="count"></span> <button id="export">导出 labels.json</button></div>
<div id="items"></div>
<script>
const ITEMS = __ITEMS__;
const KEY = "lilia-addressee-labels-v1";
let state = JSON.parse(localStorage.getItem(KEY) || "{}");
const save = () => localStorage.setItem(KEY, JSON.stringify(state));
function render() {
  const box = document.getElementById("items"); box.innerHTML = "";
  let done = 0;
  for (const item of ITEMS) {
    const key = item.chat_id + ":" + item.message_id;
    const mark = state[key];
    if (mark !== undefined) done++;
    const d = document.createElement("div");
    d.className = "item" + (mark === true ? " her" : mark === false ? " not" : "");
    d.innerHTML = item.lines.map(l => `<div class="line${l.target ? " target" : ""}${l.self ? " self" : ""}">${l.text}</div>`).join("") +
      `<div style="margin-top:6px"><button data-v="1">对她说</button><button data-v="0">不是对她</button><button data-v="">清除</button></div>`;
    d.querySelectorAll("button").forEach(b => b.onclick = () => {
      if (b.dataset.v === "") delete state[key]; else state[key] = b.dataset.v === "1";
      save(); render();
    });
    box.appendChild(d);
  }
  document.getElementById("count").textContent = `已标 ${done}/${ITEMS.length}`;
}
document.getElementById("export").onclick = () => {
  const labels = Object.entries(state).map(([key, to_her]) => {
    const [chat_id, message_id] = key.split(":").map(Number);
    return {chat_id, message_id, to_her};
  });
  const blob = new Blob([JSON.stringify({labels}, null, 1)], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "addressee-labels.json"; a.click();
};
render();
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--data-dir", default="~/.local/share/lilia")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = os.path.expanduser(args.data_dir)
    db = sqlite3.connect(f"file:{os.path.join(data, 'lilia.db')}?mode=ro", uri=True)
    rows = db.execute(
        "SELECT id, message_id, sender_id, sender_name, text, from_self FROM messages "
        "WHERE chat_type = 'group' AND chat_id = ? ORDER BY id",
        (args.group,),
    ).fetchall()
    index = {row[0]: i for i, row in enumerate(rows)}
    judged = {
        message_id
        for (message_id,) in db.execute(
            "SELECT message_id FROM addressee_labels WHERE chat_type = 'group' AND chat_id = ? AND source = 'judge'",
            (args.group,),
        )
    }
    # People's lines (not hers, not notices), judged ones first, newest first.
    candidates = [row for row in rows if not row[5] and row[2] != 0]
    candidates.sort(key=lambda row: (row[1] not in judged, -row[0]))
    picked = sorted(candidates[: args.count], key=lambda row: row[0])

    items = []
    for row in picked:
        at = index[row[0]]
        lines = []
        for other in rows[max(0, at - 6) : at + 4]:
            name = "莉莉娅(她)" if other[5] else (other[3] or str(other[2]))
            text = html.escape(f"{name}: {other[4]}")
            lines.append(
                {"text": text, "target": other[0] == row[0], "self": bool(other[5])}
            )
        items.append({"chat_id": args.group, "message_id": row[1], "lines": lines})
    page = PAGE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as out:
        out.write(page)
    os.chmod(args.out, 0o600)
    print(
        f"{len(items)} lines ({sum(row[1] in judged for row in picked)} judged) -> {args.out}"
    )


if __name__ == "__main__":
    main()
