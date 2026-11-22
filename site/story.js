"use strict";
// One story: who broke it, who followed and when, each outlet's headline (linked), headline edits,
// agency copy, and the outlets the selection model expected that stayed silent.

shell("Story file");
const id = new URLSearchParams(location.search).get("id") || "";

if (!/^\d+$/.test(id)) {
  $("inside").innerHTML = pendingHTML("No story chosen. Pick one from the front page.");
} else {
  getJSON(`data/stories/${id}.json`).then(render).catch(() => {
    $("inside").innerHTML = `<div class="pending"><b>Story not found</b>Story pages are kept for the last 28 days. <a href="./#all">Browse current stories</a></div>`;
  });
}

function render(S) {
  document.title = `${S.title} · Nazariya`;
  const t0 = new Date(S.first_seen).getTime();
  const rel = (iso) => {
    const h = (new Date(iso) - t0) / 3.6e6;
    return h < 0.05 ? "first" : h < 1 ? `+${Math.round(h * 60)} min` : h < 48 ? `+${h.toFixed(h < 10 ? 1 : 0)} h` : `+${(h / 24).toFixed(1)} days`;
  };
  const span = (new Date(S.last_seen) - t0) / 3.6e6;
  const langs = Object.entries(S.langs).map(([l, n]) => `${n} ${l === "hi" ? "Hindi" : l === "en" ? "English" : l}`).join(", ");
  const facts = [
    ["Outlets", S.n_outlets], ["Articles", S.n_articles], ["Story ran for", span < 1 ? `${Math.round(span * 60)} min` : span < 48 ? `${span.toFixed(1)} h` : `${(span / 24).toFixed(1)} days`],
    ["Expected", S.expected != null ? `~${Math.round(S.expected)} more` : "n/a"],
  ];
  $("inside").innerHTML = `
    <div class="story-head">
      <div class="kicker k-pink">Story file &middot; broke in ${esc(S.broke_in)}${S.broke_as ? ` as ${esc(S.broke_as)}` : ""}</div>
      <h2 class="hed big-hed">${esc(S.title)}</h2>
      <p class="dek">First reported ${when(S.first_seen)} IST. Covered by ${langs} outlet${S.n_outlets > 1 ? "s" : ""}${S.states.length ? `; naming ${S.states.slice(0, 3).map(esc).join(", ")}${S.states.length > 3 ? ` and ${S.states.length - 3} more states` : ""}` : ""}.</p>
      <div class="statline left">${facts.map(([k, v]) => `<span><b>${esc(v)}</b>${esc(k)}</span>`).join("")}</div>
    </div>
    <div class="inside-grid">
      <section class="col-main">
        <h3 class="sub">Who got there, and when</h3>
        <div class="halftone-box"><svg id="tl"></svg></div>
        <div class="legend">${legendGroups()}<span><i class="dot" style="background:${PAPER}"></i>agency copy only</span></div>
        <h3 class="sub">Every outlet's headline</h3>
        <div id="heads"></div>
      </section>
      <aside class="col-side">
        <div class="file"><h3>Stayed silent</h3><div id="silent"></div></div>
        <div class="file" style="margin-top:28px;box-shadow:6px 6px 0 var(--blue)"><h3>The file</h3><div id="facts"></div></div>
      </aside>
    </div>`;
  timeline(S);

  $("heads").innerHTML = S.outlets.map((o) => `
    <div class="head-row">
      <div class="hr-meta"><a class="tag" href="${outletHref(o.id)}" style="background:${GC[o.group]};color:${o.group === "English regional" ? INK : "#fff"}">${esc(o.name)}</a>
        <span>${rel(o.first_at)}</span>${o.wire ? `<span class="wire-badge">${o.wire === o.n ? "agency copy" : `${o.wire} of ${o.n} agency`}</span>` : ""}${o.n > 1 ? `<span>${o.n} articles</span>` : ""}</div>
      <div class="hr-title">${o.edits.length ? diffWords(o.edits[0].old, o.edits[o.edits.length - 1].new) : extLink(o.url, esc(o.title || "(no headline)"))}</div>
      ${o.edits.length ? `<div class="hr-edit">Headline changed after publishing. ${extLink(o.url, "Read the article &rarr;")}</div>` : ""}
    </div>`).join("");

  $("silent").innerHTML = S.silent.length
    ? `<p class="small muted">The model gave these outlets a good chance of covering it. They did not.</p>` +
      S.silent.map((o) => `<div class="nb"><a href="${outletHref(o.id)}"><i class="dot" style="background:${GC[o.group]}"></i>${esc(o.name)}</a><span>${pct(o.p)} likely</span></div>`).join("")
    : `<p class="small muted">${S.expected == null ? "The coverage model has not been fitted yet." : "Every outlet the model favoured covered it."}</p>`;

  const rows = [
    ["Broke in", esc(S.broke_in)], ["Languages", esc(langs)],
    ["States named", S.states.length ? S.states.map(esc).join(", ") : "none"],
    ["Largest toll in a headline", S.toll ?? "none"], ["Frame", S.frame ? esc(S.frame.replace(/_/g, " ")) : "n/a"],
    ["English-Hindi distance", S.lang_drift != null ? S.lang_drift.toFixed(3) : "covered in one language"],
  ];
  $("facts").innerHTML = rows.map(([k, v]) => `<div class="nb"><span>${k}</span><span>${v}</span></div>`).join("");
}

function timeline(S) {
  const rows = S.outlets, H = 34 + rows.length * 26, W = 760, L = 200, R = 50;
  const ts = rows.map((o) => new Date(o.first_at).getTime());
  const t0 = Math.min(...ts), t1 = Math.max(...ts, t0 + 3.6e6);
  const X = (t) => L + ((t - t0) / (t1 - t0)) * (W - L - R);
  const hours = (t1 - t0) / 3.6e6, step = hours <= 6 ? 1 : hours <= 24 ? 3 : hours <= 72 ? 12 : 24;
  let s = "";
  for (let h = 0; h <= hours + 1e-9; h += step) {
    const x = X(t0 + h * 3.6e6);
    s += `<line x1="${x}" x2="${x}" y1="20" y2="${H - 6}" stroke="${HAIR}" stroke-dasharray="2 4"/><text x="${x}" y="13" font-size="11" fill="${MUTE}" text-anchor="middle">${h ? "+" + h + " h" : "first"}</text>`;
  }
  rows.forEach((o, i) => {
    const y = 34 + i * 26, x = X(new Date(o.first_at).getTime()), allWire = o.wire && o.wire === o.n;
    s += `<a href="${outletHref(o.id)}"><text x="${L - 12}" y="${y + 4}" font-size="12.5" font-weight="600" fill="${INK}" text-anchor="end">${esc(o.name)}</text></a>`;
    s += `<line x1="${L}" x2="${x}" y1="${y}" y2="${y}" stroke="${GC[o.group]}" stroke-width="3" opacity=".35"/>`;
    s += `<circle cx="${x}" cy="${y}" r="8" fill="${allWire ? PAPER : GC[o.group]}" stroke="${INK}" stroke-width="2"><title>${esc(o.name)}: ${when(o.first_at)} IST${allWire ? ", agency copy" : ""}</title></circle>`;
    if (o.edits.length) s += `<text x="${x + 13}" y="${y + 4}" font-size="11" font-weight="700" fill="${PINK}">edited</text>`;
  });
  const tl = $("tl");
  tl.setAttribute("viewBox", `0 0 ${W} ${H}`);
  tl.innerHTML = s;
}
