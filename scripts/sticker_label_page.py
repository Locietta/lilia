"""Build an offline page for labelling which stickers fit which feelings.

    python scripts/sticker_label_page.py --data-dir ~/.local/share/lilia --out stickers-label.html

Needs Pillow. The page embeds thumbnails of the registered stickers (animated
GIFs stay animated) and a list of feelings; clicking a sticker cycles it
through good / ok / unmarked for the current feeling, and its pencil edits its
tags and description. Labels and edits persist in the browser and export as
labels.json, for `lilia retrieval --labels` and scripts/apply_sticker_edits.py.
The page holds the stickers themselves: keep it out of the tree (ops/ is
ignored).
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
MAX_FRAMES = 64


def flatten(frame):
    """A thumbnail of the frame over white, as lilia shows pictures to models."""
    copy = frame.convert("RGBA")
    copy.thumbnail((THUMB, THUMB))
    flat = Image.new("RGB", copy.size, "white")
    flat.paste(copy, mask=copy.getchannel("A"))
    return flat


def thumbnail(path):
    with Image.open(path) as image:
        if getattr(image, "is_animated", False) and image.n_frames > 1:
            # The iterator seeks one image in place: copy each frame as it
            # comes. Long animations keep every step-th frame, at the same pace.
            # Frames are opaque: Pillow loses the transparency of some RGBA
            # frames and paints it black.
            step = -(-image.n_frames // MAX_FRAMES)
            frames, durations = [], []
            for index, frame in enumerate(ImageSequence.Iterator(image)):
                duration = frame.info.get("duration") or 100
                if index % step:
                    durations[-1] += duration
                    continue
                frames.append(flatten(frame))
                durations.append(duration)
            out = io.BytesIO()
            frames[0].save(
                out,
                "GIF",
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0,
                disposal=1,
            )
            return "image/gif", out.getvalue()
        out = io.BytesIO()
        flatten(image).save(out, "JPEG", quality=85)
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
.s { width: 140px; border: 3px solid transparent; border-radius: 6px; padding: 2px; cursor: pointer; text-align: center; font-size: 11px; position: relative; }
.s img { width: 128px; height: 128px; object-fit: contain; background: #f4f4f4; }
.s.good { border-color: #2a2; background: #e6f6e6; }
.s.ok { border-color: #da2; background: #fbf3dc; }
.s .t { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #555; }
.s.edited .t { color: #15c; }
.s .e { position: absolute; top: 2px; right: 2px; border: 1px solid #bbb; background: #fff; border-radius: 4px; font-size: 12px; padding: 0 4px; cursor: pointer; opacity: .75; }
.s .e:hover { opacity: 1; }
dialog { width: 520px; }
dialog img { width: 192px; height: 192px; object-fit: contain; background: #f4f4f4; float: left; margin-right: 10px; }
dialog input, dialog textarea { width: 100%; box-sizing: border-box; margin: 4px 0 8px; font: inherit; }
dialog textarea { height: 120px; }
dialog .orig { clear: both; font-size: 12px; color: #777; }
</style></head><body>
<div id="side">
  <div><input id="newq" placeholder="添加情绪/场景" style="width: 130px"> <button id="add">添加</button></div>
  <div id="queries"></div>
  <hr><div id="editcount" style="font-size:12px;color:#15c"></div><button id="export">导出 labels.json</button>
  <label style="display:block;margin-top:6px">导入 <input type="file" id="import" accept=".json"></label>
</div>
<div id="main">
  <div id="bar"><b id="title"></b> &nbsp; 点一下 = 合适（绿），两下 = 勉强（黄），三下 = 取消。
    筛选标签：<input id="filter" placeholder="例如 开心"> <label><input type="checkbox" id="marked"> 只看已标</label></div>
  <div id="grid"></div>
</div>
<dialog id="editor"><form method="dialog">
  <img id="edimg"><b id="edid"></b>
  <div>标签（用、分隔）<input id="edtags"></div>
  <div>描述<textarea id="eddesc"></textarea></div>
  <div class="orig" id="edorig"></div>
  <button value="save">保存</button> <button value="revert">恢复原文</button> <button value="cancel">取消</button>
</form></dialog>
<script>
const STICKERS = __STICKERS__;
const DEFAULT_QUERIES = __FEELINGS__;
const KEY = "lilia-sticker-labels-v1";
let state = JSON.parse(localStorage.getItem(KEY) || "null") || {queries: DEFAULT_QUERIES.map(q => ({query: q, good: [], ok: []}))};
state.edits = state.edits || {};
let current = 0;
const save = () => localStorage.setItem(KEY, JSON.stringify(state));
// A sticker's tags and description, with the user's edits.
const view = s => state.edits[s.id] || {tags: s.tags, description: s.description};
function edit(s) {
  const v = view(s), dialog = document.getElementById("editor");
  document.getElementById("edimg").src = s.src;
  document.getElementById("edid").textContent = `s${s.id}`;
  document.getElementById("edtags").value = v.tags;
  document.getElementById("eddesc").value = v.description;
  document.getElementById("edorig").textContent = `原文：[${s.tags}] ${s.description}`;
  dialog.onclose = () => {
    if (dialog.returnValue === "save") {
      const tags = document.getElementById("edtags").value.trim(), description = document.getElementById("eddesc").value.trim();
      if (tags === s.tags && description === s.description) delete state.edits[s.id];
      else state.edits[s.id] = {tags, description};
    } else if (dialog.returnValue === "revert") delete state.edits[s.id];
    save(); render();
  };
  dialog.returnValue = "cancel"; dialog.showModal();
}
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
    const v = view(s);
    if (filter && !(v.tags + v.description).includes(filter)) continue;
    if (markedOnly && !mark) continue;
    const d = document.createElement("div"); d.className = "s " + mark + (state.edits[s.id] ? " edited" : ""); d.title = `s${s.id} [${v.tags}] ${v.description}`;
    d.innerHTML = `<img loading="lazy" src="${s.src}"><div class="t">s${s.id} ${v.tags}</div><span class="e" title="编辑标签和描述">✎</span>`;
    d.querySelector(".e").onclick = e => { e.stopPropagation(); edit(s); };
    d.onclick = () => {
      if (mark === "") q.good.push(s.id);
      else if (mark === "good") { q.good = q.good.filter(x => x !== s.id); q.ok.push(s.id); }
      else q.ok = q.ok.filter(x => x !== s.id);
      save(); render();
    };
    grid.appendChild(d);
  }
}
function render() {
  renderQueries(); renderGrid();
  const n = Object.keys(state.edits).length;
  document.getElementById("editcount").textContent = n ? `已修改 ${n} 张表情的标签/描述` : "";
}
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
  state = JSON.parse(await file.text()); state.edits = state.edits || {}; current = 0; save(); render();
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
