"""Build an offline page for labelling whom chat lines were said to.

    python scripts/addressee_label_page.py --group <id> [--count 120 | --rows 41,92,...]
        [--data-dir ~/.local/share/lilia | --db <lilia.db>] --out addressee-label.html

Picks recent lines of one group, preferring those the hindsight judge already
labelled (so `lilia feedback --import-manual` can report how often it agreed),
or exactly the given rows (messages.id, as `lilia replay --probe-db` prints
them), and shows each with the lines around it. Each line gets the judge's
choices (prompts/hindsight.md): her, another member, everyone, no one, or
unclear; the page exports labels.json ({"labels": [{chat_id, message_id,
to}]}). The page holds private chat lines: keep it out of the tree (ops/ is
ignored) and delete it when done.
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
#bar small { color: #555; }
.item { border: 1px solid #ddd; border-radius: 6px; margin: 10px 0; padding: 8px; }
.item.her { border-color: #2a2; background: #eef8ee; }
.item.other { border-color: #a22; background: #f8eeee; }
.item.group { border-color: #c80; background: #fdf5e6; }
.item.self { border-color: #88a; background: #f1f1f8; }
.item.unclear { border-color: #999; background: #f4f4f4; }
.line { font-size: 13px; color: #666; white-space: pre-wrap; }
.line.target { color: #000; font-weight: 600; background: #fff7cc; }
.line.self { color: #15c; }
button { margin-right: 6px; }
button.on { font-weight: 700; }
</style></head><body>
<div id="bar"><b>黄底那行当时是在对谁说？</b> 看前后文判断。<br>
  <small>对她：叫她、问她、接她刚说的话。对某个群友：在对另一个（或另几个）人说。对大家：对群里所有人说，没有特定对象（提问、分享、感叹给大家听）。
  不对谁：自言自语，或者只是个反应，不需要谁接。说不准：看完前后文也判断不了（不会导入）。</small><br>
  <span id="count"></span> <button id="export">导出 labels.json</button></div>
<div id="items"></div>
<script>
const ITEMS = __ITEMS__;
const KEY = "lilia-addressee-labels-v2";
const CHOICES = [["her", "对她"], ["other", "对某个群友"], ["group", "对大家"], ["self", "不对谁"], ["unclear", "说不准"]];
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
    d.className = "item" + (mark ? " " + mark : "");
    d.innerHTML = item.lines.map(l => `<div class="line${l.target ? " target" : ""}${l.self ? " self" : ""}">${l.text}</div>`).join("") +
      `<div style="margin-top:6px">` +
      CHOICES.map(([v, name]) => `<button data-v="${v}"${mark === v ? ' class="on"' : ""}>${name}</button>`).join("") +
      `<button data-v="">清除</button></div>`;
    d.querySelectorAll("button").forEach(b => b.onclick = () => {
      if (b.dataset.v === "") delete state[key]; else state[key] = b.dataset.v;
      save(); render();
    });
    box.appendChild(d);
  }
  document.getElementById("count").textContent = `已标 ${done}/${ITEMS.length}`;
}
document.getElementById("export").onclick = () => {
  const labels = ITEMS.filter(item => state[item.chat_id + ":" + item.message_id]).map(item =>
    ({chat_id: item.chat_id, message_id: item.message_id, to: state[item.chat_id + ":" + item.message_id]}));
  const blob = new Blob([JSON.stringify({labels}, null, 1)], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = `addressee-labels-${ITEMS.length ? ITEMS[0].chat_id : "none"}.json`; a.click();
};
render();
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--rows", help="comma-separated row ids to label instead")
    parser.add_argument("--data-dir", default="~/.local/share/lilia")
    parser.add_argument("--db", help="a lilia.db to read instead of the data dir's")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = os.path.expanduser(args.db or os.path.join(args.data_dir, "lilia.db"))
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
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
    if args.rows:
        wanted = {int(row) for row in args.rows.split(",")}
        picked = [row for row in candidates if row[0] in wanted]
        if len(picked) != len(wanted):
            missing = wanted - {row[0] for row in picked}
            raise SystemExit(
                f"not people's lines in group {args.group}: {sorted(missing)}"
            )
    else:
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
