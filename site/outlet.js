"use strict";
// One outlet's dossier: position among peers (and how it moved), publishing clock, states it over-
// and under-covers, big stories it skipped, stories it broke, copying and headline edits.

shell("Outlet file");
const oid = new URLSearchParams(location.search).get("id") || "";

if (!/^[a-z0-9_-]+$/i.test(oid)) {
  $("inside").innerHTML = pendingHTML("No outlet chosen. Pick one from the front page.");
} else {
  Promise.all([getJSON(`data/outlets/${oid}.json`), getJSON("data/outlets/index.json")]).then(([D, all]) => render(D, all)).catch(() => {
    $("inside").innerHTML = `<div class="pending"><b>Outlet not found</b><a href="./#map">See every outlet on the map</a></div>`;
  });
}

function render(D, all) {
  document.title = `${D.name} · Nazariya`;
  const col = GC[D.group];
  const stats = [
    ["stories covered", D.events.toLocaleString()], ["articles", D.articles.toLocaleString()],
    ["own reporting", D.own != null ? pct(D.own) : "n/a"], ["first on a story", D.first_score != null ? pct(D.first_score) : "n/a"],
    ["headline edits", D.n_edits],
  ];
  $("inside").innerHTML = `
    <div class="story-head">
      <span class="kicker" style="background:${col};color:${D.group === "English regional" ? INK : "#fff"}">${esc(D.group)} &middot; outlet file</span>
      <h2 class="hed big-hed">${esc(D.name)}</h2>
      <div class="statline left">${stats.map(([k, v]) => `<span><b>${esc(v)}</b>${esc(k)}</span>`).join("")}</div>
      <p class="owner-line">Owner: <b>${esc(D.owner || "not listed in its Wikipedia article")}</b>${D.owner_source ? ` (${extLink(D.owner_source, "source")})` : ""}${D.siblings.length ? `. Sister outlets tracked here: ${D.siblings.map((s) => `<a href="${outletHref(s.id)}">${esc(s.name)}</a>`).join(", ")}` : ""}.</p>
      <div class="pills jump">${all.map((o) => `<a class="${o.id === D.id ? "on" : ""}" href="${outletHref(o.id)}">${esc(o.name)}</a>`).join("")}</div>
    </div>
    <div class="dossier">
      <section class="d-map"><h3 class="sub">Where it sits among its peers</h3><div class="halftone-box"><svg id="mini" viewBox="0 0 520 360"></svg></div>
        <div id="nbrs"></div></section>
      <section class="d-clock"><h3 class="sub">When it publishes (IST)</h3><svg id="clock" viewBox="0 0 520 210"></svg><svg id="week" viewBox="0 0 520 90"></svg></section>
      <section class="d-states"><h3 class="sub">States it covers more, and less, than its peers</h3><div id="states"></div></section>
      <section class="d-skip"><h3 class="sub">Big stories it skipped this week</h3><p class="small muted">Stories the model gave it at least a 1 in 4 chance of covering, which it did not.</p><div id="skipped"></div></section>
      <section class="d-broke"><h3 class="sub">Stories it broke</h3><div id="broke"></div></section>
      <section class="d-copy"><h3 class="sub">Copying</h3><div id="copy"></div></section>
      <section class="d-edits"><h3 class="sub">Headlines it changed</h3><div id="edits"></div></section>
    </div>`;

  // mini map: every outlet faint, this one bold, with its weekly trail
  const pts = all.filter((o) => o.x != null);
  const svg = $("mini");
  if (!pts.length || !D.position) { svg.outerHTML = pendingHTML("Not on the map yet."); }
  else {
    const xs = pts.map((p) => p.x).concat(D.history.map((h) => h.x)), ys = pts.map((p) => p.y).concat(D.history.map((h) => h.y));
    const q = (v, p) => { const a = v.slice().sort((x, y) => x - y); return a[Math.round(p * (a.length - 1))]; };
    let [x0, x1, y0, y1] = [q(xs, 0.04), q(xs, 0.96), q(ys, 0.04), q(ys, 0.96)];
    const mx = (x1 - x0) * 0.15 || 1, my = (y1 - y0) * 0.15 || 1; [x0, x1, y0, y1] = [x0 - mx, x1 + mx, y0 - my, y1 + my];
    const cl = (v, a, b) => Math.max(a, Math.min(b, v));
    const X = (x) => 20 + ((cl(x, x0, x1) - x0) / (x1 - x0)) * 480, Y = (y) => 340 - ((cl(y, y0, y1) - y0) / (y1 - y0)) * 320;
    let s = "";
    for (const p of pts) if (p.id !== D.id) s += `<a href="${outletHref(p.id)}"><circle cx="${X(p.x)}" cy="${Y(p.y)}" r="6" fill="${GC[p.group]}" fill-opacity=".35" stroke="${INK}" stroke-opacity=".4"><title>${esc(p.name)}</title></circle></a>`;
    if (D.history.length > 1) s += `<polyline points="${D.history.map((h) => `${X(h.x)},${Y(h.y)}`).join(" ")}" fill="none" stroke="${INK}" stroke-width="2" stroke-dasharray="4 3"/>`;
    s += `<circle cx="${X(D.position.x)}" cy="${Y(D.position.y)}" r="13" fill="${col}" stroke="${INK}" stroke-width="3"/><text x="${X(D.position.x) + 18}" y="${Y(D.position.y) + 5}" font-size="14" font-weight="800" fill="${INK}" paint-order="stroke" stroke="${PAPER}" stroke-width="4">${esc(D.name)}</text>`;
    svg.innerHTML = s;
    $("nbrs").innerHTML = `<h5>Most alike</h5>` + D.position.neighbours.map((n) => `<div class="nb"><a href="${outletHref(n.id)}">${esc(n.name)}</a><span>distance ${n.d.toFixed(2)}</span></div>`).join("") +
      (D.history.length > 1 ? `<p class="small muted">Dashed line: its position week by week (${D.history.length} weeks).</p>` : `<p class="small muted">Its weekly trail appears after a second week of data.</p>`);
  }

  // publishing clock: 24 hourly bars, and the week
  const mxh = Math.max(1, ...D.clock);
  let c = "";
  D.clock.forEach((n, h) => {
    const bh = (n / mxh) * 160, x = 20 + h * 20.5;
    c += `<rect x="${x}" y="${180 - bh}" width="16" height="${bh}" fill="${h >= 22 || h < 6 ? BLUE : col}" stroke="${INK}" stroke-width="1.2"><title>${h}:00 IST: ${n} stories</title></rect>`;
    if (h % 3 === 0) c += `<text x="${x + 8}" y="198" font-size="11" fill="${MUTE}" text-anchor="middle">${h}h</text>`;
  });
  $("clock").innerHTML = c;
  const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], mxd = Math.max(1, ...D.weekdays);
  $("week").innerHTML = D.weekdays.map((n, d) => { const w = (n / mxd) * 60; return `<rect x="${20 + d * 70}" y="${62 - w}" width="56" height="${w}" fill="${d >= 5 ? PINK : YELLOW}" stroke="${INK}" stroke-width="1.2"/><text x="${48 + d * 70}" y="80" font-size="11" fill="${MUTE}" text-anchor="middle">${days[d]}</text>`; }).join("");

  const sbar = (r) => `<div class="wrow"><span>${esc(r.name)}</span><div class="bar"><i style="width:${Math.min(100, 33 * r.ratio)}%;background:${r.ratio >= 1 ? PINK : GREEN}"></i></div><span class="own">${fmtX(r.ratio)}</span></div>`;
  $("states").innerHTML = D.states_over.length
    ? `<h5>More than peers</h5>${D.states_over.filter((r) => r.ratio > 1).map(sbar).join("") || '<p class="small muted">None stands out.</p>'}<h5>Less than peers</h5>${D.states_under.map(sbar).join("") || '<p class="small muted">None stands out.</p>'}
       <p class="small muted">Its share of state-linked coverage going to each state, divided by all outlets' share. States with fewer than 10 outlet-stories are left out.</p>`
    : pendingHTML("Too few state-linked stories so far.");

  const evrow = (r, extra) => `<div class="srow">${r.page !== false ? `<a class="st" href="${storyHref(r.id)}">${esc(r.title)}</a>` : `<span class="st">${esc(r.title)}</span>`}<span class="sm">${extra}</span></div>`;
  $("skipped").innerHTML = D.skipped.length ? D.skipped.map((r) => evrow(r, `${pct(r.p)} likely &middot; ${r.n_outlets} others ran it`)).join("") : `<p class="small muted">Nothing it was likely to cover went missing this week.</p>`;
  $("broke").innerHTML = D.broke.length ? D.broke.map((r) => evrow(r, `${r.n_outlets} outlets followed`)).join("") : `<p class="small muted">No story it broke reached three outlets yet.</p>`;
  const crow = (r) => `<div class="nb"><a href="${outletHref(r.id)}">${esc(r.name)}</a><span>${r.n} ${r.n === 1 ? "story" : "stories"}</span></div>`;
  $("copy").innerHTML = (D.copies_from.length || D.copied_by.length)
    ? `<h5>Ran text first published by</h5>${D.copies_from.map(crow).join("") || '<p class="small muted">Nobody.</p>'}<h5>Its text was run by</h5>${D.copied_by.map(crow).join("") || '<p class="small muted">Nobody.</p>'}`
    : `<p class="small muted">No uncredited copying found either way.</p>`;
  $("edits").innerHTML = D.edits.length ? D.edits.map((e) => `<div class="edit"><div class="o">${e.minutes_after != null ? `changed within ${e.minutes_after < 120 ? e.minutes_after + " min" : Math.round(e.minutes_after / 60) + " h"}` : "changed"} &middot; ${extLink(e.url, "read the article")}</div><div class="h">${diffWords(e.old, e.new)}</div></div>`).join("")
    : `<p class="small muted">No changed headlines logged yet.</p>`;
}
