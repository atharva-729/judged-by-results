"""The one manual step: a yes/no page for the Phase 1 and Phase 3 spot-checks.

`jbr spotcheck` writes reports/spotcheck.html (offline, single file). Answers are
kept in the browser while you work; "Download answers" saves
spotcheck_labels.json, which goes in reports/. `jbr site` then reads it.
"""

from __future__ import annotations

import json

from jbr import config

LABELS = config.REPORTS / "spotcheck_labels.json"


def items() -> list[dict]:
    out = []
    p1 = json.loads((config.REPORTS / "spotcheck_phase1.json").read_text(encoding="utf-8"))
    for it in p1:
        out.append({"id": it["id"], "group": "Is this comment about the batter's at-bat?",
                    "context": f"Batter: {it['batter']}. Play: {it['play']}", "comment": it["comment"]})
    p3 = json.loads((config.REPORTS / "spotcheck_phase3.json").read_text(encoding="utf-8"))
    for it in p3:
        team = f" (classifier says: {it['winner']} wins)" if it.get("winner") else ""
        out.append({"id": it["id"], "group": it["question"], "context": f"Posted by a {it['fan']}{team}",
                    "comment": it["comment"]})
    return out


def precision() -> dict | None:
    """Share of 'yes' answers per check, or None if no answers yet."""
    if not LABELS.exists():
        return None
    answers = json.loads(LABELS.read_text(encoding="utf-8"))
    groups = {"attribution": "p1-", "ingame": "p3-ingame", "postgame": "p3-postgame"}
    p3 = {i["id"]: i["kind"] for i in
          json.loads((config.REPORTS / "spotcheck_phase3.json").read_text(encoding="utf-8"))}
    out = {}
    for name in groups:
        if name == "attribution":
            ids = [k for k in answers if k.startswith("p1-")]
        else:
            ids = [k for k in answers if p3.get(k) == name]
        yes = sum(answers[k] == "yes" for k in ids)
        out[name] = {"n": len(ids), "yes": yes, "precision": yes / len(ids) if ids else None}
    return out


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spot-check</title>
<style>
:root{--bg:#fcfcfb;--card:#fff;--ink:#0b0b0b;--ink2:#52514e;--line:#e3e2de;--yes:#1baf7a;--no:#e34948;--acc:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--card:#232322;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a38;--yes:#199e70;--no:#e66767;--acc:#3987e5}}
:root[data-theme="dark"]{--bg:#1a1a19;--card:#232322;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a38;--yes:#199e70;--no:#e66767;--acc:#3987e5}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--ink2);margin:0 0 20px}
.bar{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:1}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.q{font-weight:600;font-size:14px}.ctx{color:var(--ink2);font-size:13px;margin:2px 0 8px}
.c{white-space:pre-wrap;word-break:break-word;padding:8px 10px;border-left:3px solid var(--line);margin:0 0 10px}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:6px 16px;cursor:pointer}
.yes.on{background:var(--yes);border-color:var(--yes);color:#fff}.no.on{background:var(--no);border-color:var(--no);color:#fff}
.dl{background:var(--acc);border-color:var(--acc);color:#fff}
</style></head><body><main>
<h1>Spot-check (about 15 minutes)</h1>
<p class="sub">Answer yes or no for each. Answers stay in this browser while you go. When done, press
<b>Download answers</b> and put <code>spotcheck_labels.json</code> in <code>reports/</code>, then run <code>jbr site</code>.</p>
<div class="bar"><span id="count"></span><button class="dl" id="dl">Download answers</button></div>
<div id="list"></div></main>
<script id="items" type="application/json">__ITEMS__</script>
<script>
const items=JSON.parse(document.getElementById('items').textContent);
let ans={};try{ans=JSON.parse(localStorage.getItem('jbr-spotcheck')||'{}')}catch(e){}
const save=()=>{try{localStorage.setItem('jbr-spotcheck',JSON.stringify(ans))}catch(e){};count()};
const count=()=>{document.getElementById('count').textContent=Object.keys(ans).length+' of '+items.length+' answered'};
const list=document.getElementById('list');
items.forEach((it,i)=>{const d=document.createElement('div');d.className='card';
d.innerHTML=`<div class="q">${i+1}. ${it.group}</div><div class="ctx"></div><div class="c"></div>
<button class="yes">Yes</button> <button class="no">No</button>`;
d.querySelector('.ctx').textContent=it.context;d.querySelector('.c').textContent=it.comment;
const y=d.querySelector('.yes'),n=d.querySelector('.no');
const paint=()=>{y.classList.toggle('on',ans[it.id]==='yes');n.classList.toggle('on',ans[it.id]==='no')};
y.onclick=()=>{ans[it.id]='yes';paint();save()};n.onclick=()=>{ans[it.id]='no';paint();save()};paint();list.appendChild(d)});
count();
document.getElementById('dl').onclick=()=>{const b=new Blob([JSON.stringify(ans,null,2)],{type:'application/json'});
const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='spotcheck_labels.json';a.click()};
</script></body></html>
"""


def build() -> int:
    data = items()
    html = PAGE.replace("__ITEMS__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    (config.REPORTS / "spotcheck.html").write_text(html, encoding="utf-8")
    return len(data)
