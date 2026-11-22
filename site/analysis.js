"use strict";
// Renders site/data/analysis.json (written by `python -m nazariya.analytics.run`) as a front page.
// Headlines are written from the numbers. Every headline and outlet name comes from outside
// sources, so all text goes through esc().

const INK = "#1d1d1b", PAPER = "#f3eddf", HAIR = "#cfc6b2", MUTE = "#78736a";
const PINK = "#ff48b0", BLUE = "#0078bf", GREEN = "#00a95c", YELLOW = "#ffe800", ORANGE = "#ff6c2f";
const GC = { "English national": BLUE, "English regional": PINK, "Hindi": GREEN };
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 0) => (100 * x).toFixed(d) + "%";
const fmtX = (r) => (r == null ? "n/a" : r < 1 ? r.toFixed(2) + "x" : r.toFixed(1) + "x");
const hours = (d) => (d == null ? "not yet" : d * 24 < 48 ? Math.round(d * 24) + " h" : d.toFixed(1) + " days");
const ago = (iso) => {
  const m = (Date.now() - new Date(iso)) / 60000;
  return m < 90 ? Math.round(m) + " min ago" : m < 2880 ? Math.round(m / 60) + " h ago" : Math.round(m / 1440) + " days ago";
};
const pendingHTML = (msg) => `<div class="pending"><b>Collecting data</b>${esc(msg || "Not enough events yet.")}</div>`;
const pending = (el, m, what) => { if (el) el.innerHTML = pendingHTML(m?.reason || what); };
const hed = (id, html) => { const el = $(id); if (el && html) el.innerHTML = html; };
const legendGroups = () => Object.entries(GC).map(([g, c]) => `<span><i class="dot" style="background:${c}"></i>${g}</span>`).join("");
let A, MAP, NAME = {};
const nm = (id) => NAME[id] || id;
const gcol = (id) => GC[A.outlets[id]?.group] || MUTE;

fetch("data/analysis.json", { cache: "no-store" }).then((r) => r.json()).then((data) => {
  A = data;
  Object.entries(A.outlets).forEach(([id, o]) => (NAME[id] = o.name));
  for (const f of [header, under, outletMap, leadLag, selection, attention, halfLife, wire, copying, edits, translation]) {
    try { f(); } catch (e) { console.error(f.name, e); }
  }
}).catch((e) => {
  $("findings").innerHTML = `<li>The first model run has not finished yet. ${esc(e.message)}</li>`;
});

// ---------------------------------------------------------------- masthead + in brief
function header() {
  const gen = new Date(A.generated_at), since = new Date(A.data_since || A.since);
  const days = Math.max(1, Math.round((Date.now() - since) / 864e5));
  $("ed-date").textContent = gen.toLocaleDateString("en-IN", { weekday: "long", day: "numeric", month: "long", year: "numeric" });
  $("statline").innerHTML = [
    [A.events.toLocaleString(), "events"], [A.multi_outlet_events.toLocaleString(), "covered by 2+ outlets"],
    [Object.keys(A.outlets).length, "newsrooms"], [days, days > 1 ? "days of data" : "day of data"],
  ].map(([n, l]) => `<span><b>${n}</b>${l}</span>`).join("");
  $("generated").textContent = `Models updated ${ago(A.generated_at)}, from events since ${since.toLocaleDateString("en-IN", { day: "numeric", month: "short" })} (rolling ${Math.round(A.window_days)}-day window).`;
  const anchor = { selection: "lead", attention: "atlas", half_life: "life", lead_lag: "first", headline_edits: "edits", wire: "wire" };
  const tick = A.findings.map((f) => `<span>${esc(f.text)}</span>`).join("");
  if (tick) $("ticker").innerHTML = tick + tick;  // doubled so the loop is seamless
  else $("ticker").closest(".ticker").remove();
  $("findings").innerHTML = A.findings.length
    ? A.findings.map((f) => `<li><a href="#${anchor[f.section] || f.section}">${esc(f.text)}</a></li>`).join("")
    : `<li>Findings appear once the models have enough events.</li>`;
}

// ---------------------------------------------------------------- lead: passed over
function under() {
  const s = A.selection, el = $("underbox");
  if (!s.ready) return pending(el, s);
  if (!s.under_covered.length) return pending(el, null, "No event this week fell far below its predicted coverage.");
  hed("hed-lead", `${s.under_covered.length} stories the newsrooms <em>let go</em>`);
  el.innerHTML = s.under_covered.map((e) => `
    <div class="item"><div class="t">${esc(e.title)}</div>
      <div class="score"><b>${e.actual}/${Math.round(e.expected)}</b> other outlets picked it up, against the ${Math.round(e.expected)} expected</div>
      <div class="m">Broke in ${esc(e.broke_in)}, ${ago(e.first_seen)}. Covered by ${e.outlets.map(nm).map(esc).join(", ")}.</div></div>`).join("");
}

// ---------------------------------------------------------------- the map
function outletMap() {
  const m = A.outlet_map, svg = $("mapsvg");
  $("maplegend").innerHTML = legendGroups() + `<span>Dot size: events covered</span><span>Ring: how far the selected outlet moves when events are resampled</span>`;
  if (!m.ready) { svg.outerHTML = pendingHTML(m.reason); return; }
  // axis range from the 5th to 95th percentile so one outlier does not squash the rest
  const q = (v, p) => { const a = v.slice().sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.max(0, Math.round(p * (a.length - 1))))]; };
  const xs = m.points.map((p) => p.x), ys = m.points.map((p) => p.y);
  let [x0, x1, y0, y1] = [q(xs, 0.05), q(xs, 0.95), q(ys, 0.05), q(ys, 0.95)];
  const mx = (x1 - x0) * 0.15 || 1, my = (y1 - y0) * 0.15 || 1;
  [x0, x1, y0, y1] = [x0 - mx, x1 + mx, y0 - my, y1 + my];
  const W = 760, H = 520, cl = (v, a, b) => Math.max(a, Math.min(b, v));
  const X = (x) => 30 + ((cl(x, x0, x1) - x0) / (x1 - x0)) * (W - 170), Y = (y) => H - 30 - ((cl(y, y0, y1) - y0) / (y1 - y0)) * (H - 60);
  MAP = { X, Y, sxs: (W - 170) / (x1 - x0), sys: (H - 60) / (y1 - y0) };
  const maxEv = Math.max(...m.points.map((p) => p.events));
  const labelled = new Set(m.points.slice().sort((a, b) => b.events - a.events).slice(0, 14).map((p) => p.id));
  let s = `<line x1="${X(0)}" x2="${X(0)}" y1="10" y2="${H - 10}" stroke="${INK}" stroke-width=".8"/><line x1="10" x2="${W - 10}" y1="${Y(0)}" y2="${Y(0)}" stroke="${INK}" stroke-width=".8"/>`;
  s += `<ellipse id="mapell" rx="0" ry="0" fill="${YELLOW}" fill-opacity=".45" stroke="${INK}" stroke-width="1.5" stroke-dasharray="5 3"/>`;
  for (const p of m.points) {
    const r = 6 + 12 * Math.sqrt(p.events / maxEv), out = p.x < x0 || p.x > x1 || p.y < y0 || p.y > y1;
    s += `<g class="outlet" data-id="${esc(p.id)}"><title>${esc(p.name)}${out ? " (beyond the plotted range)" : ""}</title>
      <circle class="pt" cx="${X(p.x)}" cy="${Y(p.y)}" r="${r}" fill="${GC[p.group]}" fill-opacity=".92" stroke="${INK}" stroke-width="1.5" ${out ? 'stroke-dasharray="3 2"' : ""}/>
      <text class="lbl" x="${X(p.x) + r + 4}" y="${Y(p.y) + 4}" font-size="12.5" font-weight="600" fill="${INK}" paint-order="stroke" stroke="${PAPER}" stroke-width="3" ${labelled.has(p.id) ? "" : 'visibility="hidden"'}>${esc(p.name)}</text></g>`;
  }
  svg.innerHTML = s;
  svg.querySelectorAll(".outlet").forEach((g) => (g.onclick = () => profile(g.dataset.id)));
  const ax = (a, i) => `<div><b>Axis ${i + 1}</b> (${pct(m.inertia[i])} of the variation)
      <h5>Stories at one end</h5><ul>${a.low.slice(0, 4).map((t) => `<li>${esc(t)}</li>`).join("")}</ul>
      <h5>Stories at the other</h5><ul>${a.high.slice(0, 4).map((t) => `<li>${esc(t)}</li>`).join("")}</ul></div>`;
  $("axes").innerHTML = m.axes.map(ax).join("");
  $("mapmethod").textContent = `${m.outlets} outlets, ${m.events.toLocaleString()} events covered by 3 or more. Correspondence analysis; ${m.bootstrap} bootstrap resamples.`;
  profile(m.points.slice().sort((a, b) => b.events - a.events)[0].id);
}

function profile(id) {
  const p = A.outlet_map.points.find((q) => q.id === id);
  document.querySelectorAll("#mapsvg .outlet").forEach((g) => {
    g.classList.toggle("on", g.dataset.id === id);
    if (g.dataset.id === id) { g.querySelector(".lbl").setAttribute("visibility", "visible"); g.parentNode.appendChild(g); }
  });
  const ell = $("mapell");
  if (ell && MAP) {
    ell.setAttribute("cx", MAP.X(p.x)); ell.setAttribute("cy", MAP.Y(p.y));
    ell.setAttribute("rx", Math.max(10, 2 * p.sx * MAP.sxs)); ell.setAttribute("ry", Math.max(10, 2 * p.sy * MAP.sys));
  }
  const w = A.wire.ready ? A.wire.rows.find((r) => r.id === id) : null;
  const l = A.lead_lag.ready ? A.lead_lag.rank.find((r) => r.id === id) : null;
  const e = A.headline_edits.ready ? A.headline_edits.by_outlet.find((r) => r.id === id) : null;
  $("profile").innerHTML = `<span class="tag" style="background:${GC[p.group]};color:${p.group === "English regional" ? INK : "#fff"}">${esc(p.group)}</span>
    <h3>${esc(p.name)}</h3>
    <div class="kv">
      <div><small>Events covered</small><b>${p.events.toLocaleString()}</b></div>
      <div><small>Own reporting</small><b>${w ? pct(w.own) : "n/a"}</b></div>
      <div><small>First on a story</small><b>${l ? pct(l.score) : "n/a"}</b></div>
      <div><small>Headline edits</small><b>${e ? e.edits : 0}</b></div>
    </div>
    <h5>Most alike</h5>
    ${p.neighbours.map((n) => `<div class="nb"><span>${esc(n.name)}</span><span>distance ${n.d.toFixed(2)}</span></div>`).join("")}`;
}

// ---------------------------------------------------------------- first past the post
function leadLag() {
  const L = A.lead_lag;
  if (!L.ready) { pending($("rank"), null, `${L.events} timed stories so far.`); $("heat").closest("details").remove(); return; }
  const top = L.rank[0];
  hed("hed-first", `${esc(top.name)} got there first <em>${pct(top.score)}</em> of the time`);
  $("rank").innerHTML = L.rank.map((r) => `
    <div class="rrow" title="${r.n} stories; 95% interval ${pct(r.lo)} to ${pct(r.hi)}"><span>${esc(r.name)}</span>
      <div class="track"><div class="ci" style="left:${100 * r.lo}%;width:${100 * (r.hi - r.lo)}%;background:${GC[r.group]}"></div><div class="pt" style="left:calc(${100 * r.score}% - 3px)"></div></div>
      <span>${pct(r.score)}</span></div>`).join("") +
    `<div class="legend">${legendGroups()}<span>Bar: 95% interval</span></div>` +
    (L.excluded.length ? `<p class="small muted">Not timed (feeds give dates without times): ${L.excluded.map((o) => esc(o.name)).join(", ")}.</p>` : "");
  const cell = {}, deg = {};
  L.cells.forEach((c) => { cell[c.row + "|" + c.col] = c; deg[c.row] = (deg[c.row] || 0) + 1; });
  const ids = L.rank.map((r) => r.id).filter((id) => (deg[id] || 0) >= 2);
  if (ids.length < 3) { $("heat").innerHTML = pendingHTML("Too few outlet pairs share enough stories yet."); }
  else {
    let h = `<table><tr><th></th>${ids.map((c) => `<th class="c"><span>${esc(nm(c))}</span></th>`).join("")}</tr>`;
    for (const r of ids) {
      h += `<tr><th class="r">${esc(nm(r))}</th>`;
      for (const c of ids) {
        const x = cell[r + "|" + c];
        if (r === c || !x) { h += `<td style="color:${HAIR}">&middot;</td>`; continue; }
        const t = Math.max(-1, Math.min(1, x.minutes / 120));
        const bg = t >= 0 ? `rgba(0,120,191,${0.12 + 0.85 * t})` : `rgba(255,72,176,${0.12 - 0.85 * t})`;
        h += `<td style="background:${bg};color:${t > 0.45 ? "#fff" : INK}" title="${esc(nm(r))} vs ${esc(nm(c))}: median ${x.minutes} min, ${x.n} shared stories">${x.minutes > 0 ? "+" : ""}${x.minutes}</td>`;
      }
      h += "</tr>";
    }
    $("heat").innerHTML = h + `</table><div class="legend" style="justify-content:center"><span><i class="dot" style="background:${BLUE}"></i>row outlet first (minutes ahead)</span><span><i class="dot" style="background:${PINK}"></i>row outlet later</span></div>`;
  }
  $("leadmethod").textContent = `${L.events.toLocaleString()} stories with 3 or more timed outlets; each outlet's first own, dated article; 95% Wilson intervals.`;
}

// ---------------------------------------------------------------- news value
function selection() {
  const S = A.selection, box = $("groups");
  if (!S.ready) { $("forest").outerHTML = pendingHTML(S.reason); box.remove(); return; }
  const names = Object.keys(S.groups);
  const draw = (g) => {
    [...box.children].forEach((b) => b.classList.toggle("on", b.textContent === g));
    const G = S.groups[g], feats = G.features.filter((f) => f.share > 0.002);
    const lx = (v) => 420 + (Math.log(Math.max(0.05, Math.min(20, v))) / Math.log(20)) * 250;
    const rowH = 36, top = 16, Hh = top + feats.length * rowH + 30;
    let s = "";
    [0.1, 0.25, 0.5, 1, 2, 4, 10].forEach((t) => {
      s += `<line x1="${lx(t)}" x2="${lx(t)}" y1="4" y2="${Hh - 24}" stroke="${t === 1 ? INK : HAIR}" stroke-width="${t === 1 ? 2 : 1}" ${t === 1 ? "" : 'stroke-dasharray="2 4"'}/><text x="${lx(t)}" y="${Hh - 6}" fill="${MUTE}" font-size="12" text-anchor="middle">${t}x</text>`;
    });
    feats.forEach((f, i) => {
      const y = top + i * rowH, up = f.or >= 1, col = up ? PINK : GREEN, sig = f.lo > 1 || f.hi < 1;
      if (i % 2 === 0) s += `<rect x="0" y="${y - 14}" width="700" height="${rowH}" fill="${YELLOW}" fill-opacity=".12"/>`;
      s += `<text x="4" y="${y + 3}" fill="${INK}" font-size="14" font-weight="600">${esc(f.label)}</text><text x="4" y="${y + 17}" fill="${MUTE}" font-size="11">${pct(f.share, 1)} of stories</text>`;
      if (isFinite(f.lo) && isFinite(f.hi)) s += `<line x1="${lx(f.lo)}" x2="${lx(f.hi)}" y1="${y + 3}" y2="${y + 3}" stroke="${col}" stroke-width="6" stroke-linecap="round" opacity=".55"/>`;
      s += `<circle cx="${lx(f.or)}" cy="${y + 3}" r="7.5" fill="${sig ? col : PAPER}" stroke="${INK}" stroke-width="2"/>`;
      s += `<text x="${up ? lx(Math.max(f.hi || f.or, f.or)) + 12 : lx(Math.min(f.lo || f.or, f.or)) - 12}" y="${y + 7}" fill="${INK}" font-size="12.5" font-weight="700" text-anchor="${up ? "start" : "end"}">${f.or.toFixed(2)}x</text>`;
    });
    const fsvg = $("forest");
    fsvg.setAttribute("viewBox", `0 0 700 ${Hh}`);
    fsvg.innerHTML = s;
    $("fit").innerHTML = `<div><small>Outlet-story pairs</small><b>${G.pairs.toLocaleString()}</b></div><div><small>Outlets</small><b>${G.outlets.length}</b></div>
      <div><small>Base coverage rate</small><b>${pct(G.base_rate, 1)}</b></div><div><small>Held-out AUC</small><b>${G.auc.toFixed(2)}</b></div>`;
    const sig = G.features.filter((f) => f.share > 0.005 && (f.lo > 1 || f.hi < 1)).sort((a, b) => Math.abs(Math.log(b.or)) - Math.abs(Math.log(a.or)))[0];
    if (sig) hed("hed-select", `${esc(g)} desks: "${esc(sig.label.toLowerCase())}" ${sig.or >= 1 ? "multiplies" : "cuts"} the odds of coverage <em>${sig.or >= 1 ? sig.or.toFixed(1) + "x" : "to " + sig.or.toFixed(2) + "x"}</em>`);
  };
  box.innerHTML = "";
  names.forEach((n) => { const b = document.createElement("button"); b.className = "pill"; b.textContent = n; b.onclick = () => draw(n); box.appendChild(b); });
  draw(names[0]);
}

// ---------------------------------------------------------------- the atlas
function attention() {
  const M = A.attention, grid = $("grid");
  if (!M.ready) { grid.outerHTML = pendingHTML(M.reason); $("detail").remove(); return; }
  const lerp = (a, b, t) => a.map((x, i) => Math.round(x + (b[i] - x) * t));
  const colour = (r) => {
    const t = Math.max(-1, Math.min(1, Math.log(Math.max(r, 0.01)) / Math.log(4)));
    return t < 0 ? lerp([243, 237, 223], [0, 169, 92], -t) : lerp([243, 237, 223], [255, 72, 176], t);
  };
  const cells = {};
  for (const s of M.states) {
    const d = document.createElement("div"), r = s.ratio ?? 1;
    d.className = "tile" + (r < 0.45 ? " light" : "");
    d.style.background = `rgb(${colour(r)})`;
    d.style.gridColumn = s.tile[0] + 1; d.style.gridRow = s.tile[1] + 1;
    d.innerHTML = `<span>${esc(s.code)}</span><span class="v">${fmtX(s.ratio)}</span>`;
    d.title = s.name;
    d.onclick = () => show(s.code);
    grid.appendChild(d); cells[s.code] = d;
  }
  const show = (code) => {
    const s = M.states.find((x) => x.code === code);
    Object.values(cells).forEach((c) => c.classList.remove("on")); cells[code].classList.add("on");
    const by = Object.entries(s.by_group).map(([g, n]) => `<div class="nb"><span><i class="dot" style="background:${GC[g]}"></i>${esc(g)}</span><span>${n}</span></div>`).join("");
    $("detail").innerHTML = `<h3>${esc(s.name)}</h3>
      <div class="big" style="color:${(s.ratio ?? 1) >= 1 ? PINK : GREEN};-webkit-text-stroke:1.5px ${INK}">${fmtX(s.ratio)}</div>
      <small class="muted">of the coverage the model expects</small>
      <div class="kv"><div><small>Expected</small><b>${Math.round(s.expected)}</b></div><div><small>Actual</small><b>${s.pairs}</b></div>
        <div><small>Stories naming it</small><b>${s.events}</b></div><div><small>People (2011)</small><b>${(s.population / 1e6).toFixed(1)}M</b></div></div>
      <h5>By outlet group</h5>${by}
      <div class="eq">log E[coverage] = ${M.coef.intercept}<br>+ ${M.coef.log_population} &middot; log(population)<br>${M.coef.log_km_from_delhi < 0 ? "&minus;" : "+"} ${Math.abs(M.coef.log_km_from_delhi)} &middot; log(km from Delhi + 50)</div>`;
  };
  const big = M.states.filter((s) => s.population >= 1e7 && s.ratio != null).sort((a, b) => a.ratio - b.ratio);
  if (big.length) {
    hed("hed-atlas", `${esc(big[0].name)} gets <em>${fmtX(big[0].ratio)}</em> the coverage it should`);
    show(big[0].code);
  }
  $("atlasmethod").textContent = `${M.pairs.toLocaleString()} outlet-story pairs naming a state (dateline-only Delhi mentions dropped). Poisson regression; deviance explained ${M.d2}.`;
}

// ---------------------------------------------------------------- shelf life
function halfLife() {
  const H = A.half_life;
  if (!H.ready) { $("survsvg").outerHTML = pendingHTML(H.reason); $("dims").remove(); return; }
  const dims = Object.keys(H.dimensions).filter((d) => H.dimensions[d].length >= 2);
  const palette = [PINK, BLUE, GREEN, ORANGE, "#9b5de5", MUTE];
  const box = $("dims"), grid = H.grid_days, maxD = grid[grid.length - 1];
  const sx = (d) => 46 + (d / maxD) * 640, sy = (p) => 14 + (1 - p) * 280;
  const draw = (dim) => {
    [...box.children].forEach((b) => b.classList.toggle("on", b.textContent === dim));
    const series = H.dimensions[dim];
    let s = `<rect x="46" y="${sy(0.5)}" width="640" height="${sy(0) - sy(0.5)}" fill="${YELLOW}" fill-opacity=".14"/>`;
    [0, 0.25, 0.5, 0.75, 1].forEach((p) => (s += `<line x1="46" x2="686" y1="${sy(p)}" y2="${sy(p)}" stroke="${p === 0.5 ? INK : HAIR}" ${p === 0.5 ? 'stroke-width="1.5" stroke-dasharray="6 4"' : ""}/><text x="38" y="${sy(p) + 4}" fill="${MUTE}" font-size="11" text-anchor="end">${p * 100}%</text>`));
    for (let d = 0; d <= maxD; d += 1) s += `<text x="${sx(d)}" y="314" fill="${MUTE}" font-size="11" text-anchor="middle">${d}d</text>`;
    series.forEach((se, i) => {
      let path = `M${sx(0)},${sy(1)}`;
      se.curve.forEach((v, j) => { if (j) path += `H${sx(grid[j]).toFixed(1)}V${sy(v).toFixed(1)}`; });
      s += `<path d="${path}" fill="none" stroke="${palette[i]}" stroke-width="3.2"/>`;
    });
    $("survsvg").innerHTML = s;
    $("survleg").innerHTML = series.map((se, i) => `<span><i class="dot" style="background:${palette[i]}"></i>${esc(se.name)} (${se.n})</span>`).join("");
    $("hl").innerHTML = `<h5>Half-life</h5>` + series.map((se, i) =>
      `<div class="r" title="log-rank chi-square vs the rest: ${se.logrank_chi2}${se.logrank_chi2 > 3.84 ? " (p < 0.05)" : ""}"><span><i class="dot" style="background:${palette[i]}"></i>${esc(se.name)}</span><b>${hours(se.half_life_days)}</b></div>`).join("");
  };
  box.innerHTML = "";
  dims.forEach((d) => { const b = document.createElement("button"); b.className = "pill"; b.textContent = d; b.onclick = () => draw(d); box.appendChild(b); });
  draw(dims.includes("Where") ? "Where" : dims[0]);
  const w = Object.fromEntries((H.dimensions.Where || []).map((s) => [s.name, s]));
  if (w.Northeast?.half_life_days && w["Other states"]?.half_life_days)
    hed("hed-life", `Northeast stories fade in <em>${hours(w.Northeast.half_life_days)}</em>; others last ${hours(w["Other states"].half_life_days)}`);
}

// ---------------------------------------------------------------- wire desk
function wire() {
  const W = A.wire, el = $("wirebox");
  if (!W.ready) return pending(el, W);
  const worst = W.rows[W.rows.length - 1];
  hed("hed-wire", `${esc(worst.name)}: <em>${pct(1 - worst.own)}</em> agency or copied text`);
  const row = (r) => `
    <div class="wrow" title="${r.articles.toLocaleString()} articles with text; 95% interval ${pct(r.lo)} to ${pct(r.hi)}${Object.keys(r.sources).length ? "; copied from: " + Object.entries(r.sources).map(([k, v]) => `${k} ${v}`).join(", ") : ""}">
      <span>${esc(r.name)}</span>
      <div class="bar"><i style="width:${100 * r.own}%;background:${GC[r.group]}"></i></div>
      <span class="own" style="${r.own < 0.5 ? `background:${PINK};padding:0 3px` : ""}">${pct(r.own)}</span></div>`;
  const rows = [...W.rows].reverse();
  el.innerHTML = rows.slice(0, 12).map(row).join("") +
    (rows.length > 12 ? `<details class="more"><summary>All ${rows.length} outlets</summary>${rows.slice(12).map(row).join("")}</details>` : "") +
    `<div class="legend">${legendGroups()}</div>`;
}

// ---------------------------------------------------------------- copy desk
function copying() {
  const C = A.copying, el = $("copybox");
  if (!C.ready) return pending(el, null, "No uncredited outlet-to-outlet copying found yet.");
  const t = C.edges[0];
  hed("hed-copy", `${esc(nm(t.target))} ran <em>${t.n}</em> stories first published by ${esc(nm(t.source))}`);
  el.innerHTML = C.edges.slice(0, 20).map((e) => `
    <div class="crow"><span><i class="dot" style="background:${gcol(e.source)}"></i>${esc(nm(e.source))}</span><span class="arrow">&rarr;</span>
      <span><i class="dot" style="background:${gcol(e.target)}"></i>${esc(nm(e.target))}</span>
      <span class="n">${e.n} stories${e.share_of_target != null ? `, ${pct(e.share_of_target, 1)}` : ""}</span></div>`).join("") +
    `<p class="small muted">Read: the outlet on the right ran text first published by the one on the left. The percentage is the share of the right-hand outlet's articles.</p>`;
}

// ---------------------------------------------------------------- corrections
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
  if (!E.ready) { pending($("editlist"), null, "No changed headlines yet. The log starts with the scheduled runs."); $("editrate").remove(); return; }
  hed("hed-edits", `<em>${E.total.toLocaleString()}</em> headlines changed after publishing`);
  $("editlist").innerHTML = E.recent.slice(0, 20).map((e) => `
    <div class="edit"><div class="o"><span class="tag" style="background:${gcol(e.outlet_id)};color:#fff">${esc(e.outlet)}</span>
      ${e.minutes_after != null ? ` changed within ${e.minutes_after < 120 ? e.minutes_after + " min" : Math.round(e.minutes_after / 60) + " h"} of publishing` : ""} &middot; <a href="${esc(e.url)}" target="_blank" rel="noopener">article</a></div>
      <div class="h">${diffWords(e.old, e.new)}</div></div>`).join("");
  $("editrate").innerHTML = `<h5 style="margin-top:4px">Per 1,000 articles</h5>` + E.by_outlet.slice(0, 15).map((r) =>
    `<div class="nb"><span>${esc(r.name)}</span><span>${r.per_1000 ?? "n/a"} (${r.edits})</span></div>`).join("");
}

// ---------------------------------------------------------------- translation
function translation() {
  const T = A.translation, el = $("transbox");
  if (!T.ready) return pending(el, T);
  el.innerHTML = `<p class="small muted" style="margin-top:0">${T.events.toLocaleString()} stories covered in both languages. Median distance ${T.median.toFixed(3)}.</p>` +
    T.top.slice(0, 5).map((e) => `<div class="titem"><div class="t">${esc(e.title)}</div>
      <div class="m"><b>${e.drift.toFixed(3)}</b>, further apart than ${e.percentile}% of stories &middot; ${e.en} English, ${e.hi} Hindi outlets</div></div>`).join("");
}
