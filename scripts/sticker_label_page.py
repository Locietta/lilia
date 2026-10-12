"""Build an offline page for labelling which stickers fit which feelings.

    python scripts/sticker_label_page.py --data-dir ~/.local/share/lilia --out stickers-label.html

Needs Pillow. The page embeds thumbnails of the registered stickers (animated
GIFs stay animated) and a list of feelings; clicking a sticker cycles it
through good / ok / unmarked for the current feeling. Labels persist in the
browser and export as labels.json for `lilia retrieval --labels`. The page holds
the stickers themselves: keep it out of the tree (ops/ is ignored).
"""

import argparse
import base64
import io
import json
import os
import sqlite3

from PIL import Image, ImageSequence

FEELINGS = [
    "无语",
    "笑死",
    "生气",
    "委屈",
    "惊讶",
    "疑惑",
    "尴尬",
    "害羞",
    "嫌弃",
    "感动",
    "破防",
    "摆烂",
    "累了",
    "饿了",
    "期待",
    "得意",
    "嘲讽",
    "安慰",
    "道歉",
    "谢谢",
    "点赞",
    "加油",
    "拒绝",
    "催促",
    "吃瓜",
    "求饶",
    "打招呼",
    "早安",
    "晚安",
    "好耶",
    "抽卡又歪了",
    "考试挂了",
    "周末一起打游戏",
]

THUMB = 128


def thumbnail(path):
    with Image.open(path) as image:
        if getattr(image, "is_animated", False) and image.n_frames > 1:
            frames, durations = [], []
            for frame in list(ImageSequence.Iterator(image))[:48]:
                copy = frame.convert("RGBA")
                copy.thumbnail((THUMB, THUMB))
                frames.append(copy)
                durations.append(frame.info.get("duration", 80))
            out = io.BytesIO()
            frames[0].save(
                out,
                "GIF",
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0,
                disposal=2,
            )
            return "image/gif", out.getvalue()
        copy = image.convert("RGBA")
        copy.thumbnail((THUMB, THUMB))
        flat = Image.new("RGB", copy.size, "white")
        flat.paste(copy, mask=copy.getchannel("A"))
        out = io.BytesIO()
        flat.save(out, "JPEG", quality=85)
        return "image/jpeg", out.getvalue()


PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>lilia sticker labels</title>
<style>
body { font-family: system-ui, sans-serif; margin: 0; display: flex; height: 100vh; }
#side { width: 220px; overflow-y: auto; border-right: 1px solid #ccc; padding: 8px; box-sizing: border-box; }
#side button.q { display: block; width: 100%; text-align: left; margin: 2px 0; padding: 4px 6px; border: 1px solid #ddd; background: #fafafa; cursor: pointer; }
#side button.q.active { background: #333; color: #fff; }
#side button.q .n { float: right; opacity: .7; }
#main { flex: 1; overflow-y: auto; padding: 8px; }
#bar { position: sticky; top: 0; background: #fff; padding: 6px 0; border-bottom: 1px solid #eee; z-index: 1; }
#grid { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.s { width: 140px; border: 3px solid transparent; border-radius: 6px; padding: 2px; cursor: pointer; text-align: center; font-size: 11px; }
.s img { width: 128px; height: 128px; object-fit: contain; background: #f4f4f4; }
.s.good { border-color: #2a2; background: #e6f6e6; }
.s.ok { border-color: #da2; background: #fbf3dc; }
.s .t { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #555; }
</style></head><body>
<div id="side">
  <div><input id="newq" placeholder="添加情绪/场景" style="width: 130px"> <button id="add">添加</button></div>
  <div id="queries"></div>
  <hr><button id="export">导出 labels.json</button>
  <label style="display:block;margin-top:6px">导入 <input type="file" id="import" accept=".json"></label>
</div>
<div id="main">
  <div id="bar"><b id="title"></b> &nbsp; 点一下 = 合适（绿），两下 = 勉强（黄），三下 = 取消。
    筛选标签：<input id="filter" placeholder="例如 开心"> <label><input type="checkbox" id="marked"> 只看已标</label></div>
  <div id="grid"></div>
</div>
<script>
const STICKERS = __STICKERS__;
const DEFAULT_QUERIES = __FEELINGS__;
const KEY = "lilia-sticker-labels-v1";
let state = JSON.parse(localStorage.getItem(KEY) || "null") || {queries: DEFAULT_QUERIES.map(q => ({query: q, good: [], ok: []}))};
let current = 0;
const save = () => localStorage.setItem(KEY, JSON.stringify(state));
function renderQueries() {
  const box = document.getElementById("queries"); box.innerHTML = "";
  state.queries.forEach((q, i) => {
    const b = document.createElement("button"); b.className = "q" + (i === current ? " active" : "");
    b.innerHTML = `${q.query}<span class="n">${q.good.length}/${q.ok.length}</span>`;
    b.onclick = () => { current = i; render(); }; box.appendChild(b);
  });
}
function renderGrid() {
  const q = state.queries[current]; document.getElementById("title").textContent = q ? q.query : "";
  const grid = document.getElementById("grid"); grid.innerHTML = "";
  if (!q) return;
  const filter = document.getElementById("filter").value.trim();
  const markedOnly = document.getElementById("marked").checked;
  for (const s of STICKERS) {
    const mark = q.good.includes(s.id) ? "good" : q.ok.includes(s.id) ? "ok" : "";
    if (filter && !(s.tags + s.description).includes(filter)) continue;
    if (markedOnly && !mark) continue;
    const d = document.createElement("div"); d.className = "s " + mark; d.title = `s${s.id} [${s.tags}] ${s.description}`;
    d.innerHTML = `<img loading="lazy" src="${s.src}"><div class="t">s${s.id} ${s.tags}</div>`;
    d.onclick = () => {
      if (mark === "") q.good.push(s.id);
      else if (mark === "good") { q.good = q.good.filter(x => x !== s.id); q.ok.push(s.id); }
      else q.ok = q.ok.filter(x => x !== s.id);
      save(); render();
    };
    grid.appendChild(d);
  }
}
function render() { renderQueries(); renderGrid(); }
document.getElementById("add").onclick = () => {
  const v = document.getElementById("newq").value.trim();
  if (v && !state.queries.some(q => q.query === v)) { state.queries.push({query: v, good: [], ok: []}); current = state.queries.length - 1; save(); render(); }
  document.getElementById("newq").value = "";
};
document.getElementById("filter").oninput = renderGrid;
document.getElementById("marked").onchange = renderGrid;
document.getElementById("export").onclick = () => {
  const blob = new Blob([JSON.stringify(state, null, 1)], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "labels.json"; a.click();
};
document.getElementById("import").onchange = async e => {
  const file = e.target.files[0]; if (!file) return;
  state = JSON.parse(await file.text()); current = 0; save(); render();
};
render();
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="~/.local/share/lilia")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = os.path.expanduser(args.data_dir)
    db = sqlite3.connect(f"file:{os.path.join(data, 'lilia.db')}?mode=ro", uri=True)
    stickers = []
    for sticker_id, path, tags, description in db.execute(
        "SELECT id, path, tags, description FROM stickers WHERE status = 'registered' ORDER BY id"
    ):
        try:
            mime, blob = thumbnail(os.path.join(data, path))
        except OSError:
            continue
        stickers.append(
            {
                "id": sticker_id,
                "tags": tags,
                "description": description,
                "src": f"data:{mime};base64,{base64.b64encode(blob).decode()}",
            }
        )
    page = PAGE.replace(
        "__STICKERS__", json.dumps(stickers, ensure_ascii=False)
    ).replace("__FEELINGS__", json.dumps(FEELINGS, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as out:
        out.write(page)
    os.chmod(args.out, 0o600)
    print(
        f"{len(stickers)} stickers, {len(FEELINGS)} feelings -> {args.out} ({len(page) / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
