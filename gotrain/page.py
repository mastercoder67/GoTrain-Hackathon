"""The GoTrain web page: chat with the agent and browse the model registry.

One self-contained document. With embedded state it is a saved diary; served by
`gotrain ui` it polls the live app. All text is inserted with textContent.
"""

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">__REFRESH__
<title>GoTrain</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#060405;--side:rgba(17,9,12,.82);--glass:rgba(28,15,19,.66);--card:#150c0f;--bubble:#22121a;--line:rgba(255,255,255,.08);
--line-hi:rgba(196,52,88,.55);--ink:#f7eef0;--soft:#dccbcf;--muted:#9c8a8f;--wine:#800020;--wine-hi:#b0284f;--rose:#f0a9bb;
--grad:linear-gradient(135deg,#c4345a 0%,#800020 55%,#4a0613 100%);--mono:"JetBrains Mono",ui-monospace,Consolas,monospace}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;display:flex;overflow:hidden}
body::before{content:"";position:fixed;inset:-20%;z-index:-1;pointer-events:none;
background:radial-gradient(40% 35% at 78% 8%,rgba(128,0,32,.35),transparent 70%),radial-gradient(35% 40% at 10% 95%,rgba(176,40,79,.18),transparent 70%)}
button{font:inherit;color:inherit;cursor:pointer}
::-webkit-scrollbar{width:10px}::-webkit-scrollbar-thumb{background:rgba(255,255,255,.08);border-radius:10px;border:3px solid transparent;background-clip:padding-box}
/* sidebar */
aside{width:276px;flex:none;background:var(--side);backdrop-filter:blur(18px);border-right:1px solid var(--line);display:flex;flex-direction:column;padding:16px 12px}
.brand{display:flex;align-items:center;gap:11px;padding:4px 8px 18px;font-weight:700;font-size:18px;letter-spacing:-.01em}
.brand small{display:block;font-size:11px;font-weight:500;color:var(--muted);letter-spacing:.06em;text-transform:uppercase}
.orb{width:34px;height:34px;border-radius:11px;background:var(--grad);display:grid;place-items:center;font-size:17px;box-shadow:0 6px 24px rgba(176,40,79,.45),inset 0 1px 0 rgba(255,255,255,.25)}
.new{display:flex;align-items:center;justify-content:center;gap:8px;width:100%;background:var(--grad);border:0;border-radius:12px;padding:11px 12px;margin-bottom:14px;font-weight:600;box-shadow:0 8px 24px rgba(128,0,32,.35);transition:transform .15s,box-shadow .15s}
.new:hover{transform:translateY(-1px);box-shadow:0 12px 30px rgba(128,0,32,.5)}
.nav button{display:flex;align-items:center;gap:10px;width:100%;background:none;border:0;border-radius:10px;padding:9px 12px;color:var(--soft);text-align:left;transition:background .15s}
.nav button:hover{background:rgba(255,255,255,.04)}.nav button.on{background:rgba(176,40,79,.16);color:var(--ink);box-shadow:inset 2px 0 0 var(--wine-hi)}
.label{color:var(--muted);font-size:11px;padding:16px 12px 6px;text-transform:uppercase;letter-spacing:.1em;font-weight:600}
.hist{overflow:auto;flex:1;margin:0 -4px;padding:0 4px}
.hist button{display:flex;align-items:center;gap:10px;width:100%;text-align:left;background:none;border:0;border-radius:10px;padding:8px 10px;color:var(--ink)}
.hist button:hover{background:rgba(255,255,255,.04)}.hist button.on{background:rgba(176,40,79,.16)}
.hist .t{flex:1;min-width:0}.hist .t span{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:14px}
.hist small{display:block;color:var(--muted);font-size:11.5px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--muted);flex:none}.dot.live{background:var(--wine-hi);box-shadow:0 0 0 0 rgba(196,52,90,.6);animation:ping 1.6s infinite}
@keyframes ping{70%{box-shadow:0 0 0 8px rgba(196,52,90,0)}100%{box-shadow:0 0 0 0 rgba(196,52,90,0)}}
.foot{color:var(--muted);font-size:11.5px;padding:12px 10px 2px;border-top:1px solid var(--line);display:flex;gap:8px}
/* main */
main{flex:1;display:flex;flex-direction:column;min-width:0;height:100vh}
.top{display:flex;align-items:center;gap:14px;padding:12px 24px;border-bottom:1px solid var(--line);min-height:60px;background:rgba(6,4,5,.55);backdrop-filter:blur(14px)}
.menu{display:none;background:none;border:0;font-size:20px}
.top h1{font-size:15px;margin:0;font-weight:600;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:flex;align-items:center;gap:10px}
.pill{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;border-radius:999px;padding:3px 10px;border:1px solid var(--line)}
.pill.live{border-color:var(--line-hi);color:var(--rose);background:rgba(176,40,79,.12)}
.meters{display:flex;gap:18px}.meter{width:150px;font-size:11.5px;color:var(--muted);font-variant-numeric:tabular-nums}
.bar{height:6px;background:rgba(255,255,255,.07);border-radius:4px;overflow:hidden;margin-top:4px}.bar i{display:block;height:100%;background:var(--grad);border-radius:4px;transition:width .6s}
#scroll{flex:1;overflow:auto;scroll-behavior:smooth}.thread{max-width:800px;margin:0 auto;padding:30px 24px 48px}
/* messages */
.row{display:flex;gap:14px;margin:0 0 24px}.row.fresh{animation:rise .35s ease both}@keyframes rise{from{opacity:0;transform:translateY(8px)}}
.row.user{justify-content:flex-end}
.user .text{background:linear-gradient(180deg,#2a141c,#200f15);border:1px solid var(--line);border-radius:22px 22px 6px 22px;padding:12px 18px;max-width:80%;white-space:pre-wrap;box-shadow:0 6px 20px rgba(0,0,0,.25)}
.avatar{width:34px;height:34px;flex:none;border-radius:50%;background:var(--grad);display:grid;place-items:center;font-size:16px;margin-top:1px;box-shadow:0 4px 16px rgba(128,0,32,.4),inset 0 1px 0 rgba(255,255,255,.2)}
.body{flex:1;min-width:0}.body>b{display:block;font-weight:600;letter-spacing:-.005em}
.body p{margin:5px 0 0;color:var(--soft);white-space:pre-wrap;overflow-wrap:anywhere}.body ul{margin:7px 0 0;padding-left:20px;color:var(--soft)}.body li{margin:3px 0}
.tone-kept .avatar,.tone-ok .avatar{box-shadow:0 0 0 2px #060405,0 0 0 4px var(--wine-hi),0 4px 18px rgba(176,40,79,.6)}
.tone-rejected .avatar{background:#2a2023;box-shadow:none;filter:grayscale(.4)}
.tone-goal>.body>b{background:linear-gradient(90deg,#fff,var(--rose));-webkit-background-clip:text;background-clip:text;color:transparent}
.badge{display:inline-block;font-size:10.5px;font-weight:700;letter-spacing:.09em;border-radius:6px;padding:2px 8px;margin-left:9px;vertical-align:2px}
.badge.kept{background:var(--grad);color:#fff;box-shadow:0 2px 10px rgba(176,40,79,.45)}.badge.rejected{border:1px solid rgba(255,255,255,.18);color:var(--muted)}
figure{margin:14px 0 0;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:10px}
figure img{width:100%;border-radius:9px;image-rendering:pixelated;display:block}figcaption{color:var(--muted);font-size:12px;margin-top:8px}
.live .avatar{animation:glow 1.6s infinite}@keyframes glow{50%{box-shadow:0 0 0 8px rgba(176,40,79,.18),0 4px 22px rgba(176,40,79,.6)}}
.shimmer{background:linear-gradient(90deg,var(--soft) 0%,#fff 45%,var(--soft) 90%);background-size:200% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;animation:sh 2s linear infinite}
@keyframes sh{to{background-position:-200% 0}}
.md p{margin:6px 0 0;white-space:normal}.md p.bullet{padding-left:16px;position:relative}.md p.bullet::before{content:"";position:absolute;left:3px;top:.7em;width:5px;height:5px;border-radius:50%;background:var(--wine-hi)}
.md strong{color:var(--ink);font-weight:600}.md code{font-family:var(--mono);font-size:12.5px;background:rgba(255,255,255,.06);border-radius:5px;padding:1px 5px}
.md table{display:block;overflow-x:auto;margin:10px 0;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:4px 10px}.md td,.md th{white-space:normal;font-size:12.5px}.md td:first-child{white-space:nowrap}
.card{background:var(--glass);border:1px solid var(--line);border-radius:16px;padding:16px 18px;margin-top:12px;backdrop-filter:blur(10px)}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:8px;border-bottom:1px solid var(--line);overflow-wrap:anywhere}th{color:var(--muted);font-weight:500;font-size:12px}
tr:last-child td{border-bottom:0}td:nth-child(2),td:nth-child(3),th:nth-child(2),th:nth-child(3){white-space:nowrap;overflow-wrap:normal}.pos{color:var(--rose)}
/* welcome */
.empty{text-align:center;padding:9vh 0 0}.empty .orb{width:68px;height:68px;font-size:32px;border-radius:22px;margin:auto;animation:float 4s ease-in-out infinite}
@keyframes float{50%{transform:translateY(-6px)}}
.empty h2{font-size:34px;line-height:1.2;margin:22px 0 8px;font-weight:700;letter-spacing:-.025em;background:linear-gradient(90deg,#fff 20%,var(--rose));-webkit-background-clip:text;background-clip:text;color:transparent}
.empty>p{color:var(--muted);margin:0 auto 30px;max-width:520px}
.chips{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;text-align:left}
.chips button{background:var(--glass);border:1px solid var(--line);border-radius:16px;padding:14px 16px;color:var(--ink);display:flex;gap:12px;align-items:flex-start;transition:transform .15s,border-color .15s,box-shadow .15s}
.chips button:hover{transform:translateY(-2px);border-color:var(--line-hi);box-shadow:0 10px 30px rgba(128,0,32,.25)}
.chips .i{font-size:20px}.chips small{display:block;color:var(--muted);font-size:12px;margin-top:2px}
.steps{display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin:26px 0 0;color:var(--muted);font-size:12.5px}.steps span{border:1px solid var(--line);border-radius:999px;padding:4px 11px}
/* composer */
.composer{padding:0 24px 20px}.box{max-width:800px;margin:0 auto;display:flex;align-items:flex-end;gap:8px;background:var(--glass);backdrop-filter:blur(16px);border:1px solid var(--line);border-radius:28px;padding:8px 8px 8px 20px;transition:border-color .2s,box-shadow .2s}
.box:focus-within{border-color:var(--line-hi);box-shadow:0 0 0 4px rgba(176,40,79,.14),0 12px 40px rgba(128,0,32,.25)}
textarea{flex:1;background:none;border:0;outline:0;resize:none;color:var(--ink);font:inherit;max-height:160px;padding:9px 0}textarea::placeholder{color:var(--muted)}
.send{width:40px;height:40px;border-radius:50%;border:0;background:var(--grad);color:#fff;font-size:18px;flex:none;box-shadow:0 4px 14px rgba(176,40,79,.5);transition:transform .15s}
.send:hover:not(:disabled){transform:scale(1.06)}.send:disabled{background:rgba(255,255,255,.08);color:var(--muted);box-shadow:none;cursor:default}
.hint{max-width:800px;margin:9px auto 0;text-align:center;color:var(--muted);font-size:11.5px}
/* registry */
.reg-head{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-bottom:22px}
.tile{background:var(--glass);border:1px solid var(--line);border-radius:16px;padding:14px 16px}.tile small{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.08em}
.tile b{display:block;font-size:20px;margin-top:2px;letter-spacing:-.01em}.tile code{font-family:var(--mono);font-size:12px;color:var(--soft);overflow-wrap:anywhere}
.task{background:var(--glass);border:1px solid var(--line);border-radius:18px;padding:18px;margin:0 0 18px;animation:rise .35s ease both}
.task header{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}.task h3{margin:0;font-size:17px;letter-spacing:-.01em;flex:1}
.chip{font-size:11px;border:1px solid var(--line);border-radius:999px;padding:2px 9px;color:var(--soft)}.chip.hot{border-color:var(--line-hi);color:var(--rose);background:rgba(176,40,79,.1)}
.data{display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:13px;color:var(--muted);background:rgba(0,0,0,.2);border:1px solid var(--line);border-radius:12px;padding:10px 12px;margin-bottom:12px}
.data span{flex:1;min-width:180px}.data code{font-family:var(--mono);color:var(--soft)}
.ver{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px 14px;padding:12px;border-radius:12px;border:1px solid transparent;transition:background .15s}
.ver:hover{background:rgba(255,255,255,.025)}.ver.best{border-color:var(--line-hi);background:rgba(176,40,79,.07)}
.ver .id{font-family:var(--mono);font-size:12.5px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.ver .meta{grid-column:1;color:var(--muted);font-size:12.5px}.ver .meta i{font-style:normal;color:var(--soft)}
.score{grid-row:1/3;grid-column:2;text-align:right;min-width:120px}.score b{font-size:18px;font-variant-numeric:tabular-nums}.score small{display:block;color:var(--muted);font-size:11px}
.mini{height:4px;background:rgba(255,255,255,.07);border-radius:3px;margin-top:5px;overflow:hidden}.mini i{display:block;height:100%;background:var(--grad)}
.acts{grid-column:1/3;display:flex;gap:8px;flex-wrap:wrap}
.btn{background:rgba(255,255,255,.04);border:1px solid var(--line);border-radius:9px;padding:5px 11px;font-size:12.5px;transition:border-color .15s,background .15s}
.btn:hover:not(:disabled){border-color:var(--line-hi);background:rgba(176,40,79,.12)}.btn:disabled{opacity:.45;cursor:default}.btn.primary{background:var(--grad);border:0}
.tag{font-family:Inter,sans-serif;font-size:10px;font-weight:700;letter-spacing:.08em;border-radius:5px;padding:1px 6px}
.tag.best{background:var(--grad);color:#fff}.tag.pub{background:rgba(240,169,187,.14);color:var(--rose)}.tag.ok{background:rgba(255,255,255,.08);color:var(--soft)}.tag.fail{border:1px solid rgba(255,255,255,.2);color:var(--muted)}
.feed{list-style:none;margin:0;padding:0}.feed li{display:flex;gap:10px;padding:8px 0;border-bottom:1px solid var(--line);font-size:13px;color:var(--soft)}.feed li:last-child{border:0}
.feed time{color:var(--muted);font-size:12px;min-width:120px;font-variant-numeric:tabular-nums}
.toast{position:fixed;right:24px;bottom:24px;max-width:380px;background:#1d0f14;border:1px solid var(--line-hi);border-radius:14px;padding:12px 16px;box-shadow:0 18px 50px rgba(0,0,0,.5);animation:rise .25s ease both;z-index:9}
.toast.bad{border-color:rgba(255,255,255,.2)}
@media (max-width:820px){aside{position:fixed;inset:0 auto 0 0;z-index:5;transform:translateX(-100%);transition:transform .2s}
aside.open{transform:none}.menu{display:block}.meters{display:none}.chips,.reg-head{grid-template-columns:1fr}.thread,.composer{padding-left:16px;padding-right:16px}.top{padding:12px 16px}}
</style></head><body>
<aside id="side"><div class="brand"><span class="orb">🛰️</span><div>GoTrain<small>Geospatial AI scientist</small></div></div>
<button class="new" id="new">＋ New training request</button>
<div class="nav"><button id="nav-chat" class="on">💬 Chat</button><button id="nav-reg">🗄️ Model registry</button></div>
<div class="label">Sessions</div><div class="hist" id="hist"></div>
<div class="foot">🔒 Validation split only. The test split is never opened in a session.</div></aside>
<main><div class="top"><button class="menu" id="menu" aria-label="Sessions">☰</button><h1 id="title">GoTrain</h1>
<div class="meters" id="meters"></div></div>
<div id="scroll"><div class="thread" id="thread"></div></div>
<div class="composer" id="composer"><div class="box"><textarea id="input" rows="1"
placeholder="Ask GoTrain to train a model… e.g. segment solar panels in 30 minutes"></textarea>
<button class="send" id="send" aria-label="Send">↑</button></div>
<div class="hint" id="hint">GoTrain trains on the team GPU, one request at a time. Every score shown is a measured validation score.</div></div></main>
<script id="state" type="application/json">__STATE__</script>
<script>
const saved = JSON.parse(document.getElementById("state").textContent);
const live = saved === null;
let current = saved ? saved.state.session_id : null, chat = [], lastJson = "", view = "chat", lastSessions = [], shown = {key: null, count: 0};
const $ = id => document.getElementById(id);
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
const fmtTime = t => t ? new Date(t).toLocaleString([], {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"}) : "";
const SUGGEST = [
  ["☀️", "Train the best solar-panel segmentation model in 30 minutes", "Where are the panels? · m-pv4ger-seg"],
  ["🐄", "Find cattle in aerial photos, 45 minutes, 3 experiments", "Pixel masks · m-nz-cattle"],
  ["🔎", "Is there a solar panel? Train a classifier in 40 minutes", "Yes / no · m-pv4ger"],
  ["🌍", "Improve land-cover classification on EuroSAT in 30 minutes", "10 classes, 13 bands · m-eurosat"]];

function toast(text, bad) { const t = el("div", "toast" + (bad ? " bad" : ""), text); document.body.append(t); setTimeout(() => t.remove(), 4200); }
function agentRow(m) {
  const row = el("div", "row agent tone-" + m.tone); row.append(el("div", "avatar", m.icon));
  const body = el("div", "body"), title = el("b", null, m.title);
  if (m.tone === "kept" || m.tone === "rejected") title.append(el("span", "badge " + m.tone, m.tone === "kept" ? "KEPT" : "REJECTED"));
  body.append(title);
  if (Array.isArray(m.detail)) { const ul = el("ul"); m.detail.forEach(d => ul.append(el("li", null, d))); body.append(ul); }
  else if (m.detail) body.append(/\*\*|^\s*\|/m.test(m.detail) ? richText(m.detail) : el("p", null, m.detail));
  (m.images || []).forEach(i => { const f = el("figure"), img = el("img"); img.src = i.src; img.alt = i.caption; img.loading = "lazy"; f.append(img, el("figcaption", null, i.caption)); body.append(f); });
  row.append(body); return row;
}
/* The agent's own summary is markdown: render bold, code, bullets and tables as DOM nodes, never HTML. */
function inline(node, text) {
  text.split(/(\*\*[^*]+\*\*|`[^`]+`)/).forEach(part => {
    if (/^\*\*[^*]+\*\*$/.test(part)) node.append(el("strong", null, part.slice(2, -2)));
    else if (/^`[^`]+`$/.test(part)) node.append(el("code", null, part.slice(1, -1)));
    else if (part) node.append(document.createTextNode(part));
  });
}
function richText(text) {
  const box = el("div", "md"), lines = text.split("\n"), isRow = l => /^\s*\|.*\|\s*$/.test(l);
  for (let i = 0; i < lines.length;) {
    if (isRow(lines[i])) {
      const table = el("table"); let head = true;
      for (; i < lines.length && isRow(lines[i]); i++) {
        const cells = lines[i].trim().slice(1, -1).split("|").map(c => c.trim());
        if (cells.every(c => /^:?-{2,}:?$/.test(c))) continue;
        const tr = el("tr"); cells.forEach(c => { const cell = el(head ? "th" : "td"); inline(cell, c); tr.append(cell); });
        table.append(tr); head = false;
      }
      box.append(table); continue;
    }
    const line = lines[i++];
    if (!line.trim()) continue;
    const bullet = /^\s*[-*] /.test(line), para = el("p", bullet ? "bullet" : null);
    inline(para, line.replace(/^\s*[-*] /, "")); box.append(para);
  }
  return box;
}
function userRow(text) { const row = el("div", "row user"); row.append(el("div", "text", text)); return row; }
function delta(v) { return v == null ? "—" : (v >= 0 ? "+" : "") + v.toFixed(4); }
function memoryCard(mem) {
  if (!mem || !mem.scoreboard.length) return null;
  const row = el("div", "row agent tone-goal"); row.append(el("div", "avatar", "🧠"));
  const body = el("div", "body"); body.append(el("b", null, "What GoTrain has learned so far"));
  body.append(el("p", null, mem.lessons + " lessons from " + mem.tasks + " task(s). Kept = beat the best validation score at the time."));
  const card = el("div", "card"), t = el("table"), head = el("tr");
  ["Change", "Kept", "Mean Δ", "Tasks"].forEach(h => head.append(el("th", null, h))); t.append(head);
  mem.scoreboard.forEach(b => { const r = el("tr");
    r.append(el("td", null, b.change), el("td", null, b.kept + " / " + b.tried), el("td", b.mean_delta > 0 ? "pos" : null, delta(b.mean_delta)), el("td", null, b.tasks.join(", ")));
    t.append(r); });
  card.append(t); body.append(card); row.append(body); return row;
}
function meter(label, used, total) {
  const m = el("div", "meter", label + " · " + Math.round(used) + " / " + total), bar = el("div", "bar"), fill = el("i");
  fill.style.width = Math.min(100, 100 * used / total) + "%"; bar.append(fill); m.append(bar); return m;
}
function setTitle(text, pill) { const h = $("title"); h.replaceChildren(document.createTextNode(text)); if (pill) h.append(el("span", "pill " + (pill === "Live" ? "live" : ""), pill)); }
function renderEmpty() {
  setTitle("New request"); $("meters").replaceChildren();
  const box = el("div", "empty"); box.append(el("div", "orb", "🛰️"), el("h2", null, "What should GoTrain learn today?"),
    el("p", null, "Name a dataset and a time budget. GoTrain inspects the data, trains, diagnoses its own mistakes, tests a fix and keeps only what measurably works."));
  const chips = el("div", "chips");
  SUGGEST.forEach(([icon, text, sub]) => { const b = el("button"), t = el("div"); t.append(text, el("small", null, sub)); b.append(el("span", "i", icon), t); b.onclick = () => send(text); chips.append(b); });
  const steps = el("div", "steps"); ["🔍 Inspect", "🧪 Hypothesize", "🔬 Diagnose", "⚖️ Keep or reject", "📚 Remember"].forEach(s => steps.append(el("span", null, s)));
  box.append(chips, steps); $("thread").replaceChildren(box);
}
function render(data) {
  if (data.sessions) { lastSessions = data.sessions; renderSessions(); }
  if (view !== "chat") return;
  const s = data.state;
  if (!s) { if (chat.length) { setTitle("New request"); $("meters").replaceChildren(); $("thread").replaceChildren(...chat.map((c, i) => { const r = c.role === "user" ? userRow(c.text) : agentRow(c); if (i === chat.length - 1) r.classList.add("fresh"); return r; })); } else renderEmpty(); return; }
  setTitle(s.task, s.done ? "Finished" : "Live");
  $("meters").replaceChildren(meter("Minutes", s.used_minutes, s.minutes), meter("Experiments", s.attempts, s.experiments));
  const scroll = $("scroll"), atBottom = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 120;
  const rows = [userRow(s.request), ...s.messages.map(agentRow)];
  if (s.live) { const r = agentRow({icon: "🛰️", title: s.live.title, detail: s.live.detail, tone: "info"}); r.classList.add("live"); r.querySelector("b").classList.add("shimmer"); rows.push(r); }
  const mem = memoryCard(s.memory); if (mem && s.done) rows.push(mem);
  const key = s.session_id, before = shown.key === key ? shown.count : 0;
  rows.forEach((r, i) => { if (i >= before) r.classList.add("fresh"); });
  shown = {key, count: rows.length - (s.live ? 1 : 0)};
  $("thread").replaceChildren(...rows);
  if (atBottom) scroll.scrollTop = scroll.scrollHeight;
}
function renderSessions() {
  const hist = $("hist"); hist.replaceChildren();
  if (!lastSessions.length) { hist.append(el("div", "label", "No sessions yet")); return; }
  lastSessions.forEach((x, n) => { const b = el("button", x.session_id === current && view === "chat" ? "on" : null), t = el("div", "t");
    b.append(el("span", "dot" + (x.live ? " live" : "")));
    t.append(el("span", null, x.task), el("small", null, fmtTime(x.started_at))); b.append(t);
    b.onclick = () => { if (!live) return; view = "chat"; nav(); current = x.session_id; chat = []; lastJson = ""; $("side").classList.remove("open"); poll(); };
    hist.append(b); });
}
async function poll() {
  if (!live) return;
  if (view === "registry") return loadRegistry();
  try { const r = await fetch("api/state" + (current ? "?session=" + encodeURIComponent(current) : ""));
    const text = await r.text(); if (text !== lastJson) { lastJson = text; render(JSON.parse(text)); } } catch (e) {}
}
async function send(text) {
  text = (text || $("input").value).trim(); if (!text) return;
  $("input").value = ""; resize();
  if (!live) { toast("This is a saved diary. Start new requests from the live app: python -m gotrain ui"); return; }
  view = "chat"; nav(); current = null; chat.push({role: "user", text}); render({state: null});
  try { const r = await fetch("api/start", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({message: text})});
    const reply = await r.json();
    if (reply.started) { chat.push({icon: "🛰️", title: "On it: " + reply.task + ", " + reply.experiments + " experiments, " + Math.round(reply.minutes) + " minutes.", detail: "Loading the data and connecting to Claude…", tone: "goal", images: []}); render({state: null}); await waitForSession(); }
    else { chat.push({icon: "🛰️", title: reply.reply, detail: reply.detail || "", tone: "info", images: []}); render({state: null}); }
  } catch (e) { chat.push({icon: "⚠️", title: "Could not reach GoTrain", detail: String(e), tone: "rejected", images: []}); render({state: null}); }
}
async function waitForSession() {
  for (let i = 0; i < 80 && !current; i++) {
    const j = await (await fetch("api/launch")).json();
    if (j.session_id) { current = j.session_id; break; }
    if (j.error) { chat.push({icon: "⚠️", title: "The agent stopped before starting", detail: j.error, tone: "rejected", images: []}); render({state: null}); return; }
    await new Promise(r => setTimeout(r, 1500));
  }
  chat = []; lastJson = ""; poll();
}
/* registry */
function nav() {
  $("nav-chat").className = view === "chat" ? "on" : ""; $("nav-reg").className = view === "registry" ? "on" : "";
  $("composer").style.display = view === "chat" ? "" : "none"; renderSessions();
}
async function loadRegistry() {
  try { const text = await (await fetch("api/registry")).text(); if (text !== lastJson) { lastJson = text; const reg = JSON.parse(text); lastSessions = reg.sessions || lastSessions; renderSessions(); renderRegistry(reg); } } catch (e) {}
}
async function act(button, action, task, run_id) {
  const label = button.textContent; button.disabled = true; button.textContent = "Working…";
  try { const r = await fetch("api/registry", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({action, task, run_id})});
    const j = await r.json();
    if (j.ok) toast("✓ " + {"publish-model": "Model published", "restore-model": "Model restored, every checksum verified", "publish-data": "Dataset published", "restore-data": "Dataset restored and verified"}[action] + (j.result.s3_uri ? " · " + j.result.s3_uri : ""));
    else toast("✗ " + j.error, true);
  } catch (e) { toast("✗ " + e, true); }
  button.disabled = false; button.textContent = label; lastJson = ""; loadRegistry();
}
function tile(label, value, sub) { const t = el("div", "tile"); t.append(el("small", null, label), el("b", null, value)); if (sub) t.append(el("code", null, sub)); return t; }
function renderRegistry(reg) {
  setTitle("Model registry", "Versioned");
  const all = reg.tasks.flatMap(t => t.versions), published = all.filter(v => v.published).length;
  $("meters").replaceChildren();
  const head = el("div", "reg-head");
  head.append(tile("Model versions", String(all.length), all.filter(v => v.status === "completed").length + " completed"),
    tile("Published", String(published), "…/" + reg.storage.filesystem.split(/[\\/]/).filter(Boolean).slice(-2).join("/")),
    tile("Verified restores", String(all.filter(v => v.verified_at).length), "every checksum matched"));
  const parts = [head];
  reg.tasks.forEach(t => {
    const card = el("section", "task"), h = el("header");
    h.append(el("h3", null, t.task), el("span", "chip", t.kind), el("span", "chip", t.metric === "mean_iou" ? "mean IoU" : "accuracy"));
    if (t.frozen) h.append(el("span", "chip hot", "🔒 frozen for test"));
    if (t.test_score != null) h.append(el("span", "chip hot", "test " + t.test_score.toFixed(4)));
    card.append(h);
    const data = el("div", "data"), info = el("span");
    info.append("📦 Dataset: ", t.dataset_sha256 ? el("code", null, "sha256 " + t.dataset_sha256.slice(0, 12) + "…") : document.createTextNode(t.dataset_ready ? "downloaded, not published" : "not downloaded"));
    const pd = el("button", "btn", "Publish dataset"), rd = el("button", "btn", "Restore & verify");
    pd.disabled = !t.dataset_ready; rd.disabled = !t.dataset_sha256;
    pd.onclick = () => act(pd, "publish-data", t.task); rd.onclick = () => act(rd, "restore-data", t.task);
    data.append(info, pd, rd); card.append(data);
    if (!t.versions.length) card.append(el("p", "hint", "No versions yet. Ask GoTrain to train this task in the chat."));
    const best = Math.max(...t.versions.map(v => v.score || 0), 0.0001);
    t.versions.slice().reverse().forEach(v => {
      const row = el("div", "ver" + (v.best ? " best" : "")), id = el("div", "id");
      id.append(document.createTextNode(v.run_id));
      if (v.best) id.append(el("span", "tag best", "BEST"));
      if (v.published) id.append(el("span", "tag pub", "PUBLISHED"));
      if (v.verified_at) id.append(el("span", "tag ok", "VERIFIED ✓"));
      if (v.status !== "completed") id.append(el("span", "tag fail", (v.status || "unknown").toUpperCase()));
      const meta = el("div", "meta"); meta.append(el("i", null, v.hypothesis || ""), document.createElement("br"), v.recipe + " · " + fmtTime(v.created_at));
      if (v.parent) meta.append(document.createElement("br"), "↳ derived from " + v.parent);
      const score = el("div", "score");
      if (v.score != null) { score.append(el("b", null, v.score.toFixed(4)), el("small", null, "validation")); const m = el("div", "mini"), f = el("i"); f.style.width = (100 * v.score / best) + "%"; m.append(f); score.append(m); }
      const acts = el("div", "acts"), pub = el("button", "btn" + (v.best && !v.published ? " primary" : ""), v.published ? "Re-publish" : "Publish"), res = el("button", "btn", "Restore & verify");
      pub.disabled = v.status !== "completed"; res.disabled = !v.published;
      pub.onclick = () => act(pub, "publish-model", t.task, v.run_id); res.onclick = () => act(res, "restore-model", t.task, v.run_id);
      if (v.sha256) acts.append(el("span", "chip", "sha256 " + v.sha256.slice(0, 10) + "…"));
      acts.append(pub, res); row.append(id, score, meta, acts); card.append(row);
    });
    parts.push(card);
  });
  const feed = el("section", "task"); feed.append(el("header")); feed.firstChild.append(el("h3", null, "Storage activity"));
  const ul = el("ul", "feed");
  if (!reg.activity.length) ul.append(el("li", null, "No uploads or downloads yet."));
  reg.activity.forEach(a => { const li = el("li"); li.append(el("time", null, fmtTime(a.time)), el("span", null, (a.ok ? "✓ " : "✗ ") + a.action + " · " + a.task + (a.run_id ? " · " + a.run_id : "") + (a.s3_uri ? " · " + a.s3_uri : "") + (a.error ? " · " + a.error : ""))); ul.append(li); });
  feed.append(ul); parts.push(feed);
  $("thread").replaceChildren(...parts);
}
function resize() { const t = $("input"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 160) + "px"; $("send").disabled = !t.value.trim(); }
$("input").addEventListener("input", resize);
$("input").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
$("send").onclick = () => send(); $("menu").onclick = () => $("side").classList.toggle("open");
$("new").onclick = () => { view = "chat"; nav(); current = null; chat = []; lastJson = ""; $("side").classList.remove("open"); if (live) poll(); else renderEmpty(); };
$("nav-chat").onclick = () => { view = "chat"; nav(); lastJson = ""; $("side").classList.remove("open"); live ? poll() : render(saved); };
$("nav-reg").onclick = () => { view = "registry"; nav(); lastJson = ""; $("side").classList.remove("open"); poll(); };
resize();
if (live) {
  const params = new URLSearchParams(location.search);  // shareable links: ?session=… or ?view=registry
  current = params.get("session"); if (params.get("view") === "registry") view = "registry";
  nav(); poll(); setInterval(poll, 3000);
}
else { $("nav-reg").style.display = "none"; lastSessions = saved.sessions || []; render(saved);
  $("input").placeholder = "Saved diary · start new requests from the live app (python -m gotrain ui)"; }
</script></body></html>
"""
