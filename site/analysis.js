"use strict";
// Renders site/data/analysis.json (written by `python -m nazariya.analytics.run`).
// Every headline and outlet name comes from outside sources, so all text goes through esc().

const GC = { "English national": "#3a3fd8", "English regional": "#e4572e", "Hindi": "#0e9f8a" };
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 0) => (100 * x).toFixed(d) + "%";
const fmtX = (r) => (r == null ? "n/a" : r < 1 ? r.toFixed(2) + "x" : r.toFixed(1) + "x");
const hours = (d) => (d == null ? "not yet" : d * 24 < 48 ? Math.round(d * 24) + " h" : d.toFixed(1) + " days");
const ago = (iso) => {
  const m = (Date.now() - new Date(iso)) / 60000;
  return m < 90 ? Math.round(m) + " min ago" : m < 2880 ? Math.round(m / 60) + " h ago" : Math.round(m / 1440) + " days ago";
};
const pending = (el, m, what) => { el.innerHTML = `<div class="pending"><b>Collecting data</b>${esc(m?.reason || what || "Not enough events yet.")}</div>`; };
const legendGroups = () => Object.entries(GC).map(([g, c]) => `<span><i class="dot" style="background:${c}"></i>${g}</span>`).join("");
let A, MAP, NAME = {};
const nm = (id) => NAME[id] || id;
const gcol = (id) => GC[A.outlets[id]?.group] || "#6f6a7d";

fetch("data/analysis.json", { cache: "no-store" }).then((r) => r.json()).then((data) => {
  A = data;
  Object.entries(A.outlets).forEach(([id, o]) => (NAME[id] = o.name));
  const sections = [header, under, outletMap, leadLag, selection, attention, halfLife, wire, copying, edits, translation];
  for (const f of sections) {
    try { f(); } catch (e) { console.error(f.name, e); }
  }
}).catch((e) => {
  $("kicker").textContent = "No data yet";
  $("findings").innerHTML = `<li class="muted">The first model run has not finished. ${esc(e.message)}</li>`;
});

// ---------------------------------------------------------------- header + findings
function header() {
  const since = new Date(A.data_since || A.since), days = Math.max(1, Math.round((Date.now() - since) / 864e5));
  $("kicker").textContent = `${A.events.toLocaleString()} events · ${A.multi_outlet_events.toLocaleString()} covered by 2+ outlets · ${days} day${days > 1 ? "s" : ""} of data`;
  $("generated").textContent = `Models updated ${ago(A.generated_at)}. Figures cover events since ${since.toLocaleDateString("en-IN", { day: "numeric", month: "short" })} (rolling ${Math.round(A.window_days)}-day window).`;
  $("findings").innerHTML = A.findings.length
    ? A.findings.map((f) => `<li><a href="#${{ selection: "under", attention: "atlas", half_life: "life", lead_lag: "lead", headline_edits: "edits" }[f.section] || f.section}">${esc(f.text)}</a></li>`).join("")
    : `<li class="muted">Findings appear once the models have enough events.</li>`;
}

// ---------------------------------------------------------------- 01 under-covered
function under() {
  const s = A.selection, el = $("underbox");
  if (!s.ready) return pending(el, s);
  if (!s.under_covered.length) return pending(el, null, "No event this week fell far below its predicted coverage.");
  el.innerHTML = s.under_covered.map((e) => `
    <div class="item"><div><div class="t">${esc(e.title)}</div>
      <div class="m">Broke in ${esc(e.broke_in)} · ${ago(e.first_seen)} · covered by ${e.outlets.map(nm).map(esc).join(", ")}</div></div>
      <div class="gauge"><b>${e.actual} of ~${Math.round(e.expected)}</b>other outlets, z ${e.z}</div></div>`).join("");
}

// ---------------------------------------------------------------- 02 outlet map
function outletMap() {
  const m = A.outlet_map, svg = $("mapsvg");
  $("maplegend").innerHTML = legendGroups() + `<span>Circle size: events covered</span><span>Dashed ring: how far the selected outlet moves when events are resampled</span>`;
  if (!m.ready) { svg.outerHTML = `<div class="pending"><b>Collecting data</b>${esc(m.reason)}</div>`; return; }
  // axis range from the 5th to 95th percentile, so one far outlier does not squash everyone else;
  // outlets beyond it are drawn at the edge
  const q = (v, p) => { const a = v.slice().sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.max(0, Math.round(p * (a.length - 1))))]; };
  const xs = m.points.map((p) => p.x), ys = m.points.map((p) => p.y);
  let [x0, x1, y0, y1] = [q(xs, 0.05), q(xs, 0.95), q(ys, 0.05), q(ys, 0.95)];
  const mx = (x1 - x0) * 0.15 || 1, my = (y1 - y0) * 0.15 || 1;
  [x0, x1, y0, y1] = [x0 - mx, x1 + mx, y0 - my, y1 + my];
  const W = 760, H = 520, cl = (v, a, b) => Math.max(a, Math.min(b, v));
  const X = (x) => 40 + ((cl(x, x0, x1) - x0) / (x1 - x0)) * (W - 180), Y = (y) => H - 40 - ((cl(y, y0, y1) - y0) / (y1 - y0)) * (H - 80);
  MAP = { X, Y, sxs: (W - 180) / (x1 - x0), sys: (H - 80) / (y1 - y0) };
  const maxEv = Math.max(...m.points.map((p) => p.events));
  const labelled = new Set(m.points.slice().sort((a, b) => b.events - a.events).slice(0, 14).map((p) => p.id));
  let s = `<line x1="${X(0)}" x2="${X(0)}" y1="20" y2="${H - 30}" stroke="#d9d1bf" stroke-dasharray="4 4"/><line x1="30" x2="${W - 130}" y1="${Y(0)}" y2="${Y(0)}" stroke="#d9d1bf" stroke-dasharray="4 4"/>`;
  s += `<text x="${W - 134}" y="${Y(0) - 8}" font-size="11" fill="#6f6a7d" text-anchor="end">axis 1 →</text><text x="${X(0) + 6}" y="30" font-size="11" fill="#6f6a7d">↑ axis 2</text>`;
  s += `<ellipse id="mapell" rx="0" ry="0" fill="none" stroke="#1b1a2e" stroke-dasharray="4 3" stroke-width="1.5"/>`;
  for (const p of m.points) {
    const r = 5 + 11 * Math.sqrt(p.events / maxEv), out = p.x < x0 || p.x > x1 || p.y < y0 || p.y > y1;
    s += `<g class="outlet" data-id="${esc(p.id)}"><title>${esc(p.name)}${out ? " (beyond the plotted range)" : ""}</title>
      <circle class="pt" cx="${X(p.x)}" cy="${Y(p.y)}" r="${r}" fill="${GC[p.group]}" fill-opacity=".88" stroke="#1b1a2e" stroke-width="1" ${out ? 'stroke-dasharray="2 2"' : ""}/>
      <text class="lbl" x="${X(p.x) + r + 4}" y="${Y(p.y) + 4}" font-size="12" fill="#1b1a2e" ${labelled.has(p.id) ? "" : 'visibility="hidden"'}>${esc(p.name)}</text></g>`;
  }
  svg.innerHTML = s;
  svg.querySelectorAll(".outlet").forEach((g) => (g.onclick = () => profile(g.dataset.id)));
  const ax = (a, i) => `<div class="panel"><b>Axis ${i + 1}</b> (${pct(m.inertia[i])} of the variation)
      <h5>Events at the low end</h5><ul>${a.low.slice(0, 4).map((t) => `<li>${esc(t)}</li>`).join("")}</ul>
      <h5>Events at the high end</h5><ul>${a.high.slice(0, 4).map((t) => `<li>${esc(t)}</li>`).join("")}</ul></div>`;
  $("axes").innerHTML = m.axes.map(ax).join("");
  $("mapmethod").innerHTML = `<span><b>Input</b> ${m.outlets} outlets x ${m.events.toLocaleString()} events covered by 3+ outlets</span><span><b>Method</b> correspondence analysis, ${m.bootstrap} bootstrap resamples</span>`;
  profile(m.points.slice().sort((a, b) => b.events - a.events)[0].id);
}

function profile(id) {
  const p = A.outlet_map.points.find((q) => q.id === id);
  document.querySelectorAll("#mapsvg .outlet").forEach((g) => {
    g.classList.toggle("on", g.dataset.id === id);
    if (g.dataset.id === id) g.querySelector(".lbl").setAttribute("visibility", "visible");
  });
  const ell = $("mapell");  // bootstrap spread (2 sd) of the selected outlet's position
  if (ell && MAP) {
    ell.setAttribute("cx", MAP.X(p.x)); ell.setAttribute("cy", MAP.Y(p.y));
    ell.setAttribute("rx", Math.max(8, 2 * p.sx * MAP.sxs)); ell.setAttribute("ry", Math.max(8, 2 * p.sy * MAP.sys));
  }
  const w = A.wire.ready ? A.wire.rows.find((r) => r.id === id) : null;
  const l = A.lead_lag.ready ? A.lead_lag.rank.find((r) => r.id === id) : null;
  const e = A.headline_edits.ready ? A.headline_edits.by_outlet.find((r) => r.id === id) : null;
  $("profile").innerHTML = `<span class="tag" style="background:${GC[p.group]}">${esc(p.group)}</span>
    <h3>${esc(p.name)}</h3>
    <div class="kv">
      <div><small>Events covered</small><b>${p.events.toLocaleString()}</b></div>
      <div><small>Own reporting</small><b>${w ? pct(w.own) : "n/a"}</b></div>
      <div><small>Ahead of the pack</small><b>${l ? pct(l.score) : "n/a"}</b></div>
      <div><small>Headline edits</small><b>${e ? e.edits : 0}</b></div>
    </div>
    <h5>Closest outlets by coverage</h5>
    ${p.neighbours.map((n) => `<div class="nb"><span>${esc(n.name)}</span><span>${n.d.toFixed(2)}</span></div>`).join("")}`;
}

// ---------------------------------------------------------------- 03 lead-lag
function leadLag() {
  const L = A.lead_lag;
  if (!L.ready) { pending($("heat"), null, `${L.events} timed events so far.`); pending($("rank"), null, " "); return; }
  const cell = {}, deg = {};
  L.cells.forEach((c) => { cell[c.row + "|" + c.col] = c; deg[c.row] = (deg[c.row] || 0) + 1; });
  const ids = L.rank.map((r) => r.id).filter((id) => (deg[id] || 0) >= 2);
  if (ids.length < 3) { pending($("heat"), null, "Too few outlet pairs share enough events yet; the ranking on the right uses every timed event."); }
  let h = `<table><tr><th></th>${ids.map((c) => `<th class="c"><span>${esc(nm(c))}</span></th>`).join("")}</tr>`;
  for (const r of ids) {
    h += `<tr><th class="r">${esc(nm(r))}</th>`;
    for (const c of ids) {
      const x = cell[r + "|" + c];
      if (r === c || !x) { h += `<td style="color:#c9c1ad">·</td>`; continue; }
      const t = Math.max(-1, Math.min(1, x.minutes / 120));
      const bg = t >= 0 ? `rgba(58,63,216,${0.12 + 0.8 * t})` : `rgba(242,165,65,${0.12 - 0.8 * t})`;
      h += `<td style="background:${bg};color:${t > 0.45 ? "#fff" : "#1b1a2e"}" title="${esc(nm(r))} vs ${esc(nm(c))}: median ${x.minutes} min, ${x.n} shared events">${x.minutes > 0 ? "+" : ""}${x.minutes}</td>`;
    }
    h += "</tr>";
  }
  if (ids.length >= 3) $("heat").innerHTML = h + `</table><div class="legend" style="justify-content:center;margin-top:12px"><span><i class="dot" style="background:#3a3fd8"></i>row outlet first (minutes ahead)</span><span><i class="dot" style="background:#f2a541"></i>row outlet later</span></div>`;
  $("rank").innerHTML = `<h5 style="margin-top:0">Ahead of the median outlet</h5>` + L.rank.map((r) => `
    <div class="rrow" title="${r.n} events; 95% interval ${pct(r.lo)} to ${pct(r.hi)}"><span>${esc(r.name)}</span>
      <div class="track"><div class="ci" style="left:${100 * r.lo}%;width:${100 * (r.hi - r.lo)}%;background:${GC[r.group]}"></div><div class="pt" style="left:calc(${100 * r.score}% - 2px)"></div></div>
      <span>${pct(r.score)}</span></div>`).join("") +
    (L.excluded.length ? `<p class="muted small">Not timed (feeds give dates without times): ${L.excluded.map((o) => esc(o.name)).join(", ")}.</p>` : "");
  $("leadmethod").innerHTML = `<span><b>Events</b> ${L.events.toLocaleString()} with 3+ timed outlets</span><span><b>Timing</b> each outlet's first own, dated article</span><span><b>Interval</b> 95% Wilson</span>`;
}

// ---------------------------------------------------------------- 04 selection model
function selection() {
  const S = A.selection, box = $("groups");
  if (!S.ready) { pending($("forest").parentElement, S); return; }
  const names = Object.keys(S.groups);
  const draw = (g) => {
    [...box.children].forEach((b) => b.classList.toggle("on", b.textContent === g));
    const G = S.groups[g], feats = G.features.filter((f) => f.share > 0.002);
    const lx = (v) => 600 + (Math.log(Math.max(0.05, Math.min(20, v))) / Math.log(20)) * 380;
    const rowH = 34, top = 18, Hh = top + feats.length * rowH + 30;
    let s = "";
    [0.1, 0.25, 0.5, 1, 2, 4, 10].forEach((t) => {
      s += `<line x1="${lx(t)}" x2="${lx(t)}" y1="6" y2="${Hh - 24}" stroke="${t === 1 ? "#1b1a2e" : "#e3dccd"}" ${t === 1 ? 'stroke-width="1.5"' : 'stroke-dasharray="3 4"'}/><text x="${lx(t)}" y="${Hh - 6}" fill="#6f6a7d" font-size="12" text-anchor="middle">${t}x</text>`;
    });
    feats.forEach((f, i) => {
      const y = top + i * rowH, up = f.or >= 1, col = up ? "#e4572e" : "#0e9f8a";
      const sig = (f.lo > 1 || f.hi < 1);
      s += `<text x="0" y="${y + 4}" fill="#1b1a2e" font-size="14">${esc(f.label)}</text><text x="0" y="${y + 18}" fill="#6f6a7d" font-size="11">${pct(f.share, 1)} of events</text>`;
      if (isFinite(f.lo) && isFinite(f.hi)) s += `<line x1="${lx(f.lo)}" x2="${lx(f.hi)}" y1="${y + 4}" y2="${y + 4}" stroke="${col}" stroke-width="3" stroke-linecap="round" opacity=".5"/>`;
      s += `<circle cx="${lx(f.or)}" cy="${y + 4}" r="7" fill="${sig ? col : "#fffdf8"}" stroke="${col}" stroke-width="2"/>`;
      s += `<text x="${up ? lx(Math.max(f.hi || f.or, f.or)) + 12 : lx(Math.min(f.lo || f.or, f.or)) - 12}" y="${y + 8}" fill="#3d3a4f" font-size="12" font-family="ui-monospace,Consolas,monospace" text-anchor="${up ? "start" : "end"}">${f.or.toFixed(2)}x</text>`;
    });
    const fsvg = $("forest");
    fsvg.setAttribute("viewBox", `0 0 1000 ${Hh}`);
    fsvg.innerHTML = s;
    $("fit").innerHTML = `<div><small>(outlet, event) pairs</small><b>${G.pairs.toLocaleString()}</b></div><div><small>Outlets in group</small><b>${G.outlets.length}</b></div>
      <div><small>Base coverage rate</small><b>${pct(G.base_rate, 1)}</b></div><div><small>Held-out AUC</small><b>${G.auc.toFixed(2)}</b></div>
      <div class="muted small" style="max-width:340px">Filled dots: the 95% interval excludes 1x. Hollow dots: no clear effect yet.</div>`;
  };
  box.innerHTML = "";
  names.forEach((n) => { const b = document.createElement("button"); b.className = "pill"; b.textContent = n; b.onclick = () => draw(n); box.appendChild(b); });
  draw(names[0]);
}

// ---------------------------------------------------------------- 05 attention
function attention() {
  const M = A.attention, grid = $("grid");
  if (!M.ready) { pending(grid.parentElement, M); $("detail").remove(); return; }
  const lerp = (a, b, t) => a.map((x, i) => Math.round(x + (b[i] - x) * t));
  const colour = (r) => {
    const t = Math.max(-1, Math.min(1, Math.log(Math.max(r, 0.01)) / Math.log(4)));
    return t < 0 ? lerp([246, 241, 231], [14, 159, 138], -t) : lerp([246, 241, 231], [228, 87, 46], t);
  };
  const cells = {};
  for (const s of M.states) {
    const d = document.createElement("div"), r = s.ratio ?? 1, rgb = colour(r);
    d.className = "tile" + (Math.abs(Math.log(Math.max(r, 0.01))) > Math.log(2.2) ? " light" : "");
    d.style.background = `rgb(${rgb})`;
    d.style.gridColumn = s.tile[0] + 1; d.style.gridRow = s.tile[1] + 1;
    d.innerHTML = `<span>${esc(s.code)}</span><span class="v">${fmtX(s.ratio)}</span>`;
    d.title = s.name;
    d.onclick = () => show(s.code);
    grid.appendChild(d); cells[s.code] = d;
  }
  const show = (code) => {
    const s = M.states.find((x) => x.code === code);
    Object.values(cells).forEach((c) => c.classList.remove("on")); cells[code].classList.add("on");
    const col = (s.ratio ?? 1) >= 1 ? "#e4572e" : "#0e9f8a";
    const by = Object.entries(s.by_group).map(([g, n]) => `<div class="nb"><span><i class="dot" style="background:${GC[g]}"></i>${esc(g)}</span><span>${n}</span></div>`).join("");
    $("detail").innerHTML = `<h4>${esc(s.name)}</h4><small class="muted">Actual coverage divided by the model's expectation</small>
      <div class="big" style="color:${col}">${fmtX(s.ratio)}</div>
      <div class="kv" style="margin:10px 0"><div><small>Expected</small><b>${Math.round(s.expected)}</b></div><div><small>Actual</small><b>${s.pairs}</b></div>
        <div><small>Events naming it</small><b>${s.events}</b></div><div><small>Population (2011)</small><b>${(s.population / 1e6).toFixed(1)}M</b></div></div>
      <h5>Coverage by outlet group</h5>${by}
      <div class="eq">log E[coverage] = ${M.coef.intercept}<br>+ ${M.coef.log_population} · log(population)<br>${M.coef.log_km_from_delhi < 0 ? "-" : "+"} ${Math.abs(M.coef.log_km_from_delhi)} · log(km from Delhi + 50)</div>`;
  };
  const big = M.states.filter((s) => s.population >= 1e7 && s.ratio != null);
  show(big.sort((a, b) => a.ratio - b.ratio)[0].code);
  $("atlasmethod").innerHTML = `<span><b>Outcome</b> ${M.pairs.toLocaleString()} outlet-event pairs naming a state (dateline-only Delhi mentions dropped)</span><span><b>Method</b> Poisson regression</span><span><b>Deviance explained</b> ${M.d2}</span>`;
}

// ---------------------------------------------------------------- 06 half-life
function halfLife() {
  const H = A.half_life;
  if (!H.ready) { pending($("survsvg").parentElement.parentElement, H); return; }
  const dims = Object.keys(H.dimensions).filter((d) => H.dimensions[d].length >= 2);
  const palette = ["#3a3fd8", "#e4572e", "#0e9f8a", "#f2a541", "#b0478f", "#6f6a7d"];
  const box = $("dims");
  const grid = H.grid_days, maxD = grid[grid.length - 1];
  const sx = (d) => 50 + (d / maxD) * 630, sy = (p) => 20 + (1 - p) * 270;
  const draw = (dim) => {
    [...box.children].forEach((b) => b.classList.toggle("on", b.textContent === dim));
    const series = H.dimensions[dim];
    let s = "";
    [0, 0.25, 0.5, 0.75, 1].forEach((p) => (s += `<line x1="50" x2="680" y1="${sy(p)}" y2="${sy(p)}" stroke="${p === 0.5 ? "#1b1a2e" : "#e3dccd"}" ${p === 0.5 ? 'stroke-dasharray="6 4"' : ""}/><text x="42" y="${sy(p) + 4}" fill="#6f6a7d" font-size="11" text-anchor="end">${p * 100}%</text>`));
    for (let d = 0; d <= maxD; d += 1) s += `<text x="${sx(d)}" y="312" fill="#6f6a7d" font-size="11" text-anchor="middle">${d}d</text>`;
    series.forEach((se, i) => {
      let path = `M${sx(0)},${sy(1)}`;
      se.curve.forEach((v, j) => { if (j) path += `H${sx(grid[j]).toFixed(1)}V${sy(v).toFixed(1)}`; });
      s += `<path d="${path}" fill="none" stroke="${palette[i]}" stroke-width="2.6"/>`;
    });
    $("survsvg").innerHTML = s;
    $("survleg").innerHTML = series.map((se, i) => `<span><i class="dot" style="background:${palette[i]}"></i>${esc(se.name)} (${se.n})</span>`).join("");
    $("hl").innerHTML = `<h5 style="margin-top:0">Half-life</h5>` + series.map((se, i) =>
      `<div class="r" title="log-rank chi-square vs the rest: ${se.logrank_chi2}${se.logrank_chi2 > 3.84 ? " (p < 0.05)" : ""}"><span><i class="dot" style="background:${palette[i]}"></i>${esc(se.name)}</span><b>${hours(se.half_life_days)}</b></div>`).join("") +
      `<p class="muted small">Hover a row for its log-rank test against the other groups.</p>`;
  };
  box.innerHTML = "";
  dims.forEach((d) => { const b = document.createElement("button"); b.className = "pill"; b.textContent = d; b.onclick = () => draw(d); box.appendChild(b); });
  draw(dims.includes("Where") ? "Where" : dims[0]);
}

// ---------------------------------------------------------------- 07 wire
function wire() {
  const W = A.wire, el = $("wirebox");
  if (!W.ready) return pending(el, W);
  el.innerHTML = W.rows.map((r) => `
    <div class="wrow" title="${r.articles.toLocaleString()} articles; 95% interval ${pct(r.lo)} to ${pct(r.hi)}${Object.keys(r.sources).length ? "; copied from: " + Object.entries(r.sources).map(([k, v]) => `${k} ${v}`).join(", ") : ""}">
      <span><b>${esc(r.name)}</b><br><small class="muted">${esc(r.group)}</small></span>
      <div class="bar"><i style="width:${100 * r.own}%;background:${GC[r.group]}"></i></div>
      <span class="own" style="color:${r.own < 0.5 ? "#d7263d" : "#1b1a2e"}">${pct(r.own)}</span></div>`).join("");
}

// ---------------------------------------------------------------- 08 copying
function copying() {
  const C = A.copying, el = $("copybox");
  if (!C.ready) return pending(el, null, "No uncredited outlet-to-outlet copying found yet.");
  el.innerHTML = C.edges.slice(0, 25).map((e) => `
    <div class="crow"><span><i class="dot" style="background:${gcol(e.source)}"></i>${esc(nm(e.source))}</span><span class="arrow">→</span>
      <span><i class="dot" style="background:${gcol(e.target)}"></i>${esc(nm(e.target))}</span>
      <span class="n">${e.n} articles${e.share_of_target != null ? ` · ${pct(e.share_of_target, 1)}` : ""}</span></div>`).join("") +
    `<p class="muted small">The percentage is the share of the copying outlet's articles in the window.</p>`;
}

// ---------------------------------------------------------------- 09 headline edits
function diffWords(a, b) {
  const x = a.split(/\s+/), y = b.split(/\s+/);
  const n = x.length, m = y.length, L = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = x[i] === y[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const out = []; let i = 0, j = 0;
  while (i < n && j < m) {
    if (x[i] === y[j]) { out.push(esc(x[i])); i++; j++; }
    else if (L[i + 1][j] >= L[i][j + 1]) out.push(`<del>${esc(x[i++])}</del>`);
    else out.push(`<ins>${esc(y[j++])}</ins>`);
  }
  while (i < n) out.push(`<del>${esc(x[i++])}</del>`);
  while (j < m) out.push(`<ins>${esc(y[j++])}</ins>`);
  return out.join(" ");
}

function edits() {
  const E = A.headline_edits;
  if (!E.ready) { pending($("editlist"), null, "No headline changes seen yet. The log starts with the first scheduled runs."); pending($("editrate"), null, " "); return; }
  $("editlist").innerHTML = E.recent.slice(0, 25).map((e) => `
    <div class="edit"><div class="o"><span class="tag" style="background:${gcol(e.outlet_id)}">${esc(e.outlet)}</span>
      changed ${e.minutes_after != null ? `within ${e.minutes_after < 120 ? e.minutes_after + " min" : Math.round(e.minutes_after / 60) + " h"} of publishing` : ""} · <a href="${esc(e.url)}" target="_blank" rel="noopener">article</a></div>
      <div class="h">${diffWords(e.old, e.new)}</div></div>`).join("");
  $("editrate").innerHTML = `<h5 style="margin-top:0">Edits per 1,000 articles</h5>` + E.by_outlet.slice(0, 15).map((r) =>
    `<div class="nb"><span>${esc(r.name)}</span><span>${r.per_1000 ?? "n/a"} (${r.edits})</span></div>`).join("") +
    `<p class="muted small">${E.total.toLocaleString()} edits logged in the window.</p>`;
}

// ---------------------------------------------------------------- 10 translation
function translation() {
  const T = A.translation, el = $("transbox");
  if (!T.ready) return pending(el, T);
  el.innerHTML = `<div class="muted small" style="margin-bottom:6px">${T.events.toLocaleString()} events covered in both languages. Median distance ${T.median.toFixed(3)}; the top tenth start at ${T.p90.toFixed(3)}.</div>` +
    T.top.map((e) => `<div class="item"><div><div class="t">${esc(e.title)}</div><div class="m">${e.en} English and ${e.hi} Hindi outlets</div></div>
      <div class="gauge"><b style="color:#3a3fd8">${e.drift.toFixed(3)}</b>more apart than ${e.percentile}% of events</div></div>`).join("");
}
