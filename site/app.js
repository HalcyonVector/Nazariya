"use strict";

const GROUPS = [
  { label: "English · national", test: (o) => o.role === "outlet" && o.lang === "en" && o.region === "national" },
  { label: "English · regional", test: (o) => o.role === "outlet" && o.lang === "en" && o.region !== "national" },
  { label: "Hindi", test: (o) => o.role === "outlet" && o.lang === "hi" },
  { label: "Reference", test: (o) => o.role === "reference" },
];

const FRAME_NAMES = {
  economic: "Economic", capacity_resources: "Resources", morality: "Morality", fairness_equality: "Fairness",
  legality: "Legality", policy: "Policy", crime_punishment: "Crime", security_defense: "Security",
  health_safety: "Health & safety", quality_of_life: "Quality of life", cultural_identity: "Cultural identity",
  public_opinion: "Public opinion", political: "Political", external_reputation: "Foreign relations", other: "Other",
};

function frameLine(story) {
  const entries = Object.entries(story.frames || {});
  if (!entries.length) return null;
  const line = el("p", { class: "frames", title: "Each outlet counted once, by the frame it used most on this story (automatic, provisional)" });
  line.append(el("span", { class: "label", text: "Framed as " }));
  entries.slice(0, 4).forEach(([f, n], i) => {
    if (i) line.append(" · ");
    line.append(el("b", { text: FRAME_NAMES[f] || f }), ` ${n}`);
  });
  return line;
}

const state = { stories: [], outlets: [], byId: {}, lang: "all", min: 3, q: "", entity: null };
const $ = (id) => document.getElementById(id);

function ago(iso) {
  const h = (Date.now() - new Date(iso).getTime()) / 36e5;
  if (h < 1) return `${Math.max(1, Math.round(h * 60))} min ago`;
  if (h < 48) return `${Math.round(h)} h ago`;
  return `${Math.round(h / 24)} days ago`;
}

function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else n.setAttribute(k, v);
  }
  n.append(...children.filter((c) => c != null));
  return n;
}

function strip(story) {
  const byOutlet = {};
  for (const a of story.articles) (byOutlet[a.outlet] ||= []).push(a);
  const noData = new Set(story.no_data);
  const wrap = el("div", { class: "strip" });
  for (const g of GROUPS) {
    const members = state.outlets.filter(g.test);
    if (!members.length) continue;
    const cells = el("div", { class: "cells" });
    for (const o of members) {
      const arts = byOutlet[o.id];
      let cls = "cell", tip = `${o.name}: not covered`;
      if (arts) {
        const allWire = arts.every((a) => a.syndicated || a.agency);
        cls += allWire ? " wire" : " covered";
        const fr = (story.outlet_frames || {})[o.id];
        tip = `${o.name}${fr ? ` [${FRAME_NAMES[fr] || fr}]` : ""}: ${arts[0].title}${arts.length > 1 ? ` (+${arts.length - 1} more)` : ""}${allWire ? " [wire copy]" : ""}`;
      } else if (noData.has(o.id)) {
        cls += " nodata";
        tip = `${o.name}: no data for this period`;
      }
      cells.append(el("span", { class: cls, title: tip, "aria-label": tip, role: "img" }));
    }
    wrap.append(el("div", { class: "group" }, el("span", { class: "label", text: g.label }), cells));
  }
  return wrap;
}

function articleList(story) {
  // one row per outlet: its first article on the story, plus how many follow-ups it ran
  const byOutlet = new Map();
  for (const a of story.articles) {
    if (!byOutlet.has(a.outlet)) byOutlet.set(a.outlet, []);
    byOutlet.get(a.outlet).push(a);
  }
  const list = el("ul");
  for (const [outlet, arts] of byOutlet) {
    const a = arts[0];
    const o = state.byId[outlet];
    const badges = [];
    if (a.lang === "hi") badges.push("Hindi");
    if (a.agency) badges.push(a.agency);
    else if (a.syndicated) badges.push("wire copy");
    if (a.access === "headline_only") badges.push("headline only");
    const frame = (story.outlet_frames || {})[outlet];
    if (frame) badges.push(FRAME_NAMES[frame] || frame);
    if (arts.length > 1) badges.push(`+${arts.length - 1} more`);
    const link = el("a", { href: a.url, target: "_blank", rel: "noopener noreferrer", lang: a.lang || "" }, a.title);
    list.append(el("li", {},
      el("span", { class: "outlet" }, o ? o.name : outlet, el("span", { class: "when", text: ago(a.published) })),
      el("span", {}, link, ...badges.map((b) => el("span", { class: "badge", text: b })))));
  }
  const summary = `${byOutlet.size} outlets · ${story.articles.length} articles`;
  return el("details", { class: "articles" }, el("summary", { text: summary }), list);
}

function entityChips(story) {
  if (!story.entities || !story.entities.length) return null;
  const wrap = el("div", { class: "entities" });
  for (const e of story.entities) {
    const b = el("button", { class: "ent", type: "button", title: `Show stories mentioning ${e.label}` }, e.label);
    b.addEventListener("click", () => { state.entity = { qid: e.qid, label: e.label }; render(); window.scrollTo(0, 0); });
    wrap.append(b);
  }
  return wrap;
}

function card(story) {
  const langs = story.langs.map((l) => (l === "hi" ? "Hindi" : l === "en" ? "English" : l)).join(" + ");
  return el("article", { class: "story" },
    el("h2", { text: story.title }),
    entityChips(story),
    el("div", { class: "meta" },
      el("span", { text: `${story.n_outlets} outlets` }),
      el("span", { text: `${story.n_articles} articles` }),
      el("span", { text: langs }),
      el("span", { text: `updated ${ago(story.last_seen)}` })),
    strip(story),
    frameLine(story),
    articleList(story));
}

function render() {
  const q = state.q.toLowerCase();
  const shown = state.stories.filter((s) => {
    if (s.n_outlets < state.min) return false;
    const has = new Set(s.langs);
    if (state.lang === "cross" && !(has.has("en") && has.has("hi"))) return false;
    if (state.lang === "en" && (has.has("hi") || !has.has("en"))) return false;
    if (state.lang === "hi" && (has.has("en") || !has.has("hi"))) return false;
    if (q && !s.articles.some((a) => a.title.toLowerCase().includes(q))) return false;
    if (state.entity && !(s.entities || []).some((e) => e.qid === state.entity.qid)) return false;
    return true;
  });
  const banner = $("entity-filter");
  banner.hidden = !state.entity;
  if (state.entity) $("entity-name").textContent = state.entity.label;
  $("stories").replaceChildren(...shown.map(card));
  $("status").textContent = `${shown.length} of ${state.stories.length} stories from the last ${state.meta.window_hours} hours · updated ${ago(state.meta.generated_at)}`;
}

async function main() {
  try {
    const [s, o] = await Promise.all([
      fetch("data/stories.json", { cache: "no-store" }).then((r) => r.json()),
      fetch("data/outlets.json", { cache: "no-store" }).then((r) => r.json()),
    ]);
    state.stories = s.stories;
    state.meta = s.meta;
    state.outlets = o.outlets;
    state.byId = Object.fromEntries(o.outlets.map((x) => [x.id, x]));
    render();
  } catch (e) {
    $("status").textContent = "Couldn't load data. Try again in a minute.";
  }
}

document.querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", String(c === b)));
  state.lang = b.dataset.lang;
  render();
}));
$("min").addEventListener("change", (e) => { state.min = Number(e.target.value); render(); });
$("entity-clear").addEventListener("click", () => { state.entity = null; render(); });
$("q").addEventListener("input", (e) => { state.q = e.target.value.trim(); render(); });
main();
