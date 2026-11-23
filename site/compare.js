"use strict";
// Head to head: two outlets' coverage of the stories 3+ outlets covered in the current window.

shell("Head to head");
const qs = new URLSearchParams(location.search);

Promise.all([getJSON("data/outlets/index.json"), getJSON("data/stories/index.json").catch(() => [])]).then(([all, stories]) => {
  const ids = all.map((o) => o.id);
  let a = ids.includes(qs.get("a")) ? qs.get("a") : "thehindu";
  let b = ids.includes(qs.get("b")) ? qs.get("b") : "toi";
  const story = Object.fromEntries(stories.map((s) => [s.id, s]));
  const opts = (sel) => all.map((o) => `<option value="${esc(o.id)}"${o.id === sel ? " selected" : ""}>${esc(o.name)}</option>`).join("");
  $("inside").innerHTML = `
    <div class="story-head"><div class="kicker k-pink">Head to head</div>
      <h2 class="hed big-hed" id="h2h-hed">Two newsrooms, the same news</h2>
      <p class="dek">Pick two outlets. Of the stories three or more newsrooms covered in the last four weeks: what both ran, what only one ran, who got there first, and how each headlined the stories they shared.</p>
      <div class="pickers"><select id="pa" aria-label="First outlet">${opts(a)}</select><span class="vs">vs</span><select id="pb" aria-label="Second outlet">${opts(b)}</select></div></div>
    <div id="h2h"></div>`;
  const go = () => {
    a = $("pa").value; b = $("pb").value;
    history.replaceState(null, "", `?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);
    Promise.all([getJSON(`data/outlets/${a}.json`), getJSON(`data/outlets/${b}.json`)]).then(([A, B]) => render(A, B, story));
  };
  $("pa").onchange = go; $("pb").onchange = go;
  go();
});

function render(A, B, story) {
  const ca = new Map(A.coverage.map((r) => [r[0], r])), cb = new Map(B.coverage.map((r) => [r[0], r]));
  const both = [...ca.keys()].filter((k) => cb.has(k));
  const onlyA = [...ca.keys()].filter((k) => !cb.has(k)), onlyB = [...cb.keys()].filter((k) => !ca.has(k));
  const union = ca.size + cb.size - both.length;
  const jac = union ? both.length / union : 0;
  let aFirst = 0, bFirst = 0;
  const lead = [];
  for (const k of both) {
    const d = (new Date(cb.get(k)[1]) - new Date(ca.get(k)[1])) / 60000;  // positive: A first
    if (d > 0) aFirst++; else if (d < 0) bFirst++;
    lead.push(d);
  }
  lead.sort((x, y) => x - y);
  const med = lead.length ? lead[Math.floor(lead.length / 2)] : 0;
  const size = (k) => story[k]?.n_outlets || 0;
  const title = (k, r) => story[k]?.title || r[2] || "";
  hed("h2h-hed", `${esc(A.name)} and ${esc(B.name)} shared <em>${pct(jac)}</em> of their stories`);
  const col = (D) => GC[D.group];
  const stat = (label, va, vb) => `<div class="cmp-row"><b>${va}</b><span>${label}</span><b>${vb}</b></div>`;
  const list = (keys, cov, emptyMsg) => keys.length ? keys.sort((x, y) => size(y) - size(x)).slice(0, 15).map((k) => {
    const r = cov.get(k);
    return `<div class="srow">${story[k] ? `<a class="st" href="${storyHref(k)}">${esc(title(k, r))}</a>` : `<span class="st">${esc(title(k, r))}</span>`}<span class="sm">${size(k)} outlets ran it</span></div>`;
  }).join("") : `<p class="small muted">${emptyMsg}</p>`;
  const shared = both.sort((x, y) => size(y) - size(x)).slice(0, 12).map((k) => {
    const ra = ca.get(k), rb = cb.get(k), d = Math.round((new Date(rb[1]) - new Date(ra[1])) / 60000);
    const who = d > 0 ? `${esc(A.name)} first by ${d < 120 ? d + " min" : Math.round(d / 60) + " h"}` : d < 0 ? `${esc(B.name)} first by ${-d < 120 ? -d + " min" : Math.round(-d / 60) + " h"}` : "same time";
    return `<div class="pair"><div class="pair-t">${story[k] ? `<a href="${storyHref(k)}">${esc(title(k, ra))}</a>` : esc(title(k, ra))}<small>${who}</small></div>
      <div class="pair-h"><div style="border-color:${col(A)}">${extLink(ra[3], esc(ra[2] || ""))}</div><div style="border-color:${col(B)}">${extLink(rb[3], esc(rb[2] || ""))}</div></div></div>`;
  }).join("");
  $("h2h").innerHTML = `
    <div class="cmp">
      <div class="cmp-head"><a href="${outletHref(A.id)}" style="background:${col(A)}">${esc(A.name)}</a><span></span><a href="${outletHref(B.id)}" style="background:${col(B)}">${esc(B.name)}</a></div>
      ${stat("stories covered (3+ outlets)", ca.size, cb.size)}
      ${stat("first on shared stories", aFirst, bFirst)}
      ${stat("own reporting", A.own != null ? pct(A.own) : "n/a", B.own != null ? pct(B.own) : "n/a")}
      ${stat("headline edits", A.n_edits, B.n_edits)}
      ${stat("owner", esc(A.owner || "not listed"), esc(B.owner || "not listed"))}
    </div>
    <div class="venn"><span class="v-a" style="background:${col(A)}">${onlyA.length}<small>only ${esc(A.name)}</small></span><span class="v-both">${both.length}<small>both</small></span><span class="v-b" style="background:${col(B)}">${onlyB.length}<small>only ${esc(B.name)}</small></span></div>
    <p class="small muted">On shared stories, the median gap was ${Math.abs(Math.round(med)) < 120 ? Math.abs(Math.round(med)) + " minutes" : Math.round(Math.abs(med) / 60) + " hours"} (${med > 0 ? esc(A.name) : med < 0 ? esc(B.name) : "neither"} usually first).</p>
    <div class="cmp-lists">
      <section><h3 class="sub">Only ${esc(A.name)} ran</h3>${list(onlyA, ca, "None.")}</section>
      <section><h3 class="sub">Only ${esc(B.name)} ran</h3>${list(onlyB, cb, "None.")}</section>
    </div>
    <h3 class="sub">How each headlined the stories they shared</h3>
    ${shared || `<p class="small muted">They shared no story yet.</p>`}`;
}
