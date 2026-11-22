"use strict";
// Shared by the front page, story pages and outlet dossiers. All outside text goes through esc().

const INK = "#1d1d1b", PAPER = "#f3eddf", HAIR = "#cfc6b2", MUTE = "#78736a";
const PINK = "#ff48b0", BLUE = "#0078bf", GREEN = "#00a95c", YELLOW = "#ffe800", ORANGE = "#ff6c2f";
const GC = { "English national": BLUE, "English regional": PINK, "Hindi": GREEN, "Regional language": ORANGE };
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 0) => (100 * x).toFixed(d) + "%";
const fmtX = (r) => (r == null ? "n/a" : r < 1 ? r.toFixed(2) + "x" : r.toFixed(1) + "x");
const hours = (d) => (d == null ? "not yet" : d * 24 < 48 ? Math.round(d * 24) + " h" : d.toFixed(1) + " days");
const ago = (iso) => {
  const m = (Date.now() - new Date(iso)) / 60000;
  return m < 90 ? Math.round(m) + " min ago" : m < 2880 ? Math.round(m / 60) + " h ago" : Math.round(m / 1440) + " days ago";
};
const when = (iso) => new Date(iso).toLocaleString("en-IN", { day: "numeric", month: "short", hour: "numeric", minute: "2-digit", timeZone: "Asia/Kolkata" });
const pendingHTML = (msg) => `<div class="pending"><b>Collecting data</b>${esc(msg || "Not enough events yet.")}</div>`;
const pending = (el, m, what) => { if (el) el.innerHTML = pendingHTML(m?.reason || what); };
const hed = (id, html) => { const el = $(id); if (el && html) el.innerHTML = html; };
const legendGroups = () => Object.entries(GC).map(([g, c]) => `<span><i class="dot" style="background:${c}"></i>${g}</span>`).join("");
const LANG_NAMES = { en: "English", hi: "Hindi", ta: "Tamil", bn: "Bengali", mr: "Marathi", ml: "Malayalam", te: "Telugu", kn: "Kannada" };
const langName = (l) => LANG_NAMES[l] || l;
const storyHref = (id) => `story.html?id=${encodeURIComponent(id)}`;
const outletHref = (id) => `outlet.html?id=${encodeURIComponent(id)}`;
// only http(s) links from feeds are ever rendered as links
const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? u : null);
const extLink = (u, text) => (safeUrl(u) ? `<a href="${esc(u)}" target="_blank" rel="noopener">${text}</a>` : text);
const getJSON = (path) => fetch(path, { cache: "no-store" }).then((r) => { if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`); return r.json(); });

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

const FOOTER = `<footer class="colophon">
  <div class="fcols">
    <div>
      <h5>About</h5>
      <p>Nazariya models how India's news outlets cover the same events. It measures coverage patterns, not truth or intent, and every estimate carries uncertainty. <a href="https://github.com/HalcyonVector/Nazariya" target="_blank" rel="noopener">Source code</a></p>
    </div>
    <div>
      <h5>Data</h5>
      <ul>
        <li>Outlet RSS feeds and news sitemaps: headlines, times, links (<a href="health.html" target="_blank" rel="noopener">list and status</a>)</li>
        <li><a href="https://pib.gov.in" target="_blank" rel="noopener">PIB</a>: government press releases</li>
        <li><a href="https://www.wikidata.org" target="_blank" rel="noopener">Wikidata</a>: people, parties and states named in articles</li>
        <li><a href="https://censusindia.gov.in" target="_blank" rel="noopener">Census of India 2011</a>: state populations</li>
        <li><a href="https://en.wikipedia.org" target="_blank" rel="noopener">Wikipedia</a>: who owns each outlet (cited on each outlet's file)</li>
      </ul>
    </div>
    <div>
      <h5>References</h5>
      <ul>
        <li><a href="https://en.wikipedia.org/wiki/Correspondence_analysis" target="_blank" rel="noopener">Correspondence analysis</a></li>
        <li><a href="https://en.wikipedia.org/wiki/Logistic_regression" target="_blank" rel="noopener">Logistic regression</a></li>
        <li><a href="https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator" target="_blank" rel="noopener">Kaplan-Meier estimator</a></li>
        <li><a href="https://en.wikipedia.org/wiki/Granger_causality" target="_blank" rel="noopener">Lead-lag (Granger causality)</a></li>
        <li><a href="https://arxiv.org/abs/2402.03216" target="_blank" rel="noopener">BGE M3 multilingual embeddings</a></li>
        <li><a href="https://en.wikipedia.org/wiki/Permutation_test" target="_blank" rel="noopener">Permutation test</a>, <a href="https://en.wikipedia.org/wiki/Jaccard_index" target="_blank" rel="noopener">Jaccard index</a></li>
      </ul>
    </div>
  </div>
  <div class="bottom">
    <span>Built by <b>Sagnik</b></span><i>&#10022;</i>
    <a href="https://github.com/HalcyonVector" target="_blank" rel="noopener">GitHub @HalcyonVector</a><i>&#10022;</i>
    <a href="https://github.com/HalcyonVector/Nazariya" target="_blank" rel="noopener">Repository</a><i>&#10022;</i>
    <a href="https://github.com/HalcyonVector/Nazariya/blob/main/LICENSE" target="_blank" rel="noopener">License</a>
  </div>
</footer>`;

// the masthead and footer shared by the inside pages
function shell(kicker) {
  document.body.insertAdjacentHTML("afterbegin", `<div class="paper" id="paper">
    <header class="mast small">
      <div class="edition"><a href="./">&larr; Front page</a><span class="ed-mid">${esc(kicker)}</span><a href="health.html" target="_blank" rel="noopener">Sources and feed status</a></div>
      <a class="nameplate-link" href="./"><h1 class="nameplate"><span>Nazariya</span></h1></a>
    </header>
    <main class="inside" id="inside"><p class="dek">Loading</p></main>
    ${FOOTER}</div>`);
}
