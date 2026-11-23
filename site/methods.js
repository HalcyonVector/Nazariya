"use strict";
// The methods page: how each number is made. Static text, plus a few live counts from analysis.json.

shell("Methods");

const M = [
  ["pipeline", "The pipeline", "Where the data comes from and how articles become events.",
    `<p><b>Collection.</b> Every four hours, each of 50 outlets' feeds (news sitemaps and RSS, with Google News headlines as a last resort) is read, plus the Press Information Bureau's releases and ANI's wire. robots.txt is obeyed. Items older than three days or outside the outlet's own site are dropped, and web addresses are normalised so the same article is never counted twice.</p>
     <p><b>Text.</b> Article pages are fetched newest first, up to a cap per outlet per run, and their text extracted with trafilatura. 64-bit SimHash fingerprints find re-published duplicates within an outlet and near-copies across outlets. Agency credits (PTI, ANI, IANS and others) are read from the dateline line and the page's byline metadata. Text is kept for three days and never published.</p>
     <p><b>Entities.</b> A dictionary linker over a Wikidata alias table finds people, parties and states in English and Hindi, and states in six more languages including inflected forms. On 150 articles judged by a local LLM, link precision was 0.837.</p>
     <p><b>Events.</b> The headline and first ~200 words are embedded with BGE-M3, a multilingual model, so a Hindi and an English article about the same event land close together. Each article joins the nearest open event above a cosine similarity of 0.65, or starts a new one; an event closes after 72 hours without a new article. An article that names states, none of which the event names, joins only if it is nearly a duplicate (0.80). Replayed over 20,490 articles against labelled pairs, pairwise precision was 0.872 and recall 0.841.</p>
     <p><b>Facts.</b> Each event is condensed into one record (size, timing, languages, states and people named, how the first report looked) and one record per outlet that covered it (when, how many articles, how many agency copy, its first headline and link). The models read only these; raw articles are deleted after eight days.</p>`],
  ["lead", "Stories the newsrooms passed over", "Which events got far less coverage than expected?",
    `<p>For each event of the past week, the selection model (below) gives every outlet a probability of covering it. Their sum is the expected number of other outlets; the actual count is compared with it as a z-score, treating each outlet as an independent coin with its own probability. Shown: events expected to reach at least 4 other outlets that fell at least two standard deviations short.</p>
     <p><b>Limits.</b> Outlets are not independent (they read each other), so z-scores overstate certainty somewhat. "Passed over" means the feeds did not carry the story while they were read, not that it was suppressed.</p>`],
  ["select", "What makes a story news", "Which features of a story raise or lower the odds that an outlet covers it?",
    `<p><b>Unit.</b> One row per (outlet, event) for events at least two outlets covered, plus a weighted 15% sample of single-outlet events; the outlet that broke the story is excluded. <b>Outcome:</b> whether the outlet covered it. <b>Features</b> describe the first report only (agency copy, a government release, a death toll, a named politician, a New Delhi dateline, the states named, its language and its frame), because totals over all coverage partly result from wide coverage.</p>
     <p><b>Model.</b> Logistic regression per outlet group with outlet fixed effects (absorbing each outlet's volume). Odds ratios carry 95% intervals from 25 bootstrap resamples of events. Discrimination is checked with five-fold cross-validation split by event (no event in both training and test).</p>
     <p><b>Limits.</b> Associations, not causes. Frames come from a classifier trained on LLM labels.</p>`],
  ["map", "Which newsrooms think alike", "Which outlets choose similar sets of stories?",
    `<p>Correspondence analysis of the outlet-by-event matrix (events covered by 3 or more outlets; outlets with 10 or more such events). Outlets close together chose similar events. Stability: the analysis is repeated on 40 bootstrap resamples of events, each rotated back onto the full solution (orthogonal Procrustes); the ring shows two standard deviations of the outlet's position. Neighbours use the first five dimensions. Each week's positions are stored, rotated onto the previous week's, to draw each outlet's trail.</p>
     <p><b>Limits.</b> Two dimensions capture a modest share of the variation (shown on the page); language explains much of the first axis.</p>`],
  ["first", "Who gets there first", "Which outlets publish their own report before the others?",
    `<p>For events 3 or more outlets covered, each outlet's first own article (not agency copy, with a real timestamp) is timed. Pairwise: the median minutes one outlet was ahead of another over the events both covered (15 or more). Ranking: the share of events where an outlet beat the median outlet, with 95% Wilson intervals. Outlets whose feeds give dates without times are left out.</p>`],
  ["atlas", "Coverage each state should get, and gets", "Which states get less coverage than their size and distance explain?",
    `<p><b>Outcome.</b> For each state, the number of outlet-event pairs whose articles name it (a lone "New Delhi:" dateline does not count as Delhi). <b>Model.</b> Poisson regression on log population (Census 2011) and log distance from Delhi (from Wikidata coordinates). The page shows actual divided by expected. The Northeast is deliberately not a model term, so extra neglect shows in the residuals.</p>
     <p><b>Limits.</b> 2011 populations; place is read from names in the text, so a story about a city that never names its state is missed.</p>`],
  ["life", "How long a story stays in the news", "Which kinds of stories fade fastest?",
    `<p>Kaplan-Meier survival curves of how long events keep drawing new articles (first to last article), with events still running treated as censored. Groups: where the story is, the language it broke in, its size and its frame; each group is compared with the rest by a log-rank test.</p>
     <p><b>Limits.</b> The 72-hour clustering window caps how long one event can run; a story that pauses longer starts a new event.</p>`],
  ["wire", "Own reporting, agency copy and copying", "How much of each outlet is its own work?",
    `<p>An article counts as agency copy if its text credits an agency or its byline metadata names one, or if its SimHash fingerprint is within 6 bits (of 64) of an earlier article from another outlet. Shares are over articles whose text could be read, with 95% Wilson intervals. "Who copies whom" lists near-copies with no agency credit.</p>
     <p><b>Limits.</b> Credits placed where the extractor does not look are missed, so agency copy is under-counted.</p>`],
  ["edits", "Headlines changed after publishing, and stories taken down", "Which outlets change or remove what they published?",
    `<p><b>Edits.</b> When an outlet's own feed shows a story it showed before under a different headline, both versions are kept. Ignored: differences only in punctuation, spacing or case; headlines from a different kind of feed (sitemaps and category feeds format the same headline differently); headlines rebuilt from web addresses; and live blogs, which change by design. Each edit is labelled: numbers changed, small correction, added detail or rewritten.</p>
     <p><b>Removals.</b> Each run re-checks up to 300 articles one to three days old. One counts as removed only when clearly gone (HTTP 404 or 410, or sent to the front page) on two checks at least four hours apart; blocks, rate limits and failures never count.</p>`],
  ["named", "Who gets named", "Which people and parties does each outlet put in its headlines, against its peers on the same stories?",
    `<p>For events 3 or more outlets covered, an outlet's headlines on those events are compared with its peers' headlines on the same events. For each of the eight most-named people and parties: an exact binomial test of the outlet's count against its peers' rate, with the false-discovery rate held at 5% across all outlet-name pairs (Benjamini-Hochberg). A pair is tested only if 5 or more mentions are expected.</p>
     <p><b>Limits.</b> Headlines only, the most visible choice but not the whole story. English and Hindi outlets only until people are linked in the regional languages; outlets whose headlines are rebuilt from web addresses are left out.</p>`],
  ["owners", "Who owns the news", "Do outlets with the same owner cover the same stories?",
    `<p>Owners are taken from each outlet's English Wikipedia infobox and cited on its file. Overlap of two outlets: the Jaccard index of the events (3+ outlets) each covered. The statistic is the average excess overlap of same-owner pairs over different-owner pairs in the same language combination. Significance: owners are shuffled among outlets of the same language 2,000 times.</p>
     <p><b>Limits.</b> Overlap may come from shared newsrooms or content sharing; the data shows the overlap, not its cause. Ownership changes; check the cited source.</p>`],
  ["more", "Breakout forecast, official business, clock, bridges and agenda", "",
    `<p><b>Breakout forecast.</b> Logistic regression from a story's first two hours (outlets and languages so far, how it broke, hour of day) to whether it reaches 10 or more outlets. Fitted on the older 70% of finished stories and scored on the newer 30%; the page shows that record (AUC, hit rate among its top picks, calibration).</p>
     <p><b>Official business.</b> Each PIB release is kept with the event it landed in and its ministry (from its first line). After two days it counts as reported if any tracked outlet covered that event.</p>
     <p><b>Newsroom clock.</b> Each outlet's first article on each story by hour (IST). Feeds that have only shown part of the day are left out.</p>
     <p><b>Language bridges.</b> For events 3+ outlets covered: the language group of the first report, and for each other group the first outlet to carry it and the delay.</p>
     <p><b>Agenda fingerprints.</b> Each outlet's share of stories in each frame against the share across all outlets, with 95% Wilson intervals.</p>`],
  ["labels", "Labels and what is provisional", "",
    `<p>The clustering, entity-linking and frame figures are measured against labels written by large language models (one main model, a second as a check), with a small number of human labels. They are marked provisional. A human labelling session checks the live pipeline directly: 130 yes/no questions on pairs the clustering joined, look-alikes it kept apart, and names the linker found in headlines. Its results replace the provisional clustering and linking figures once answered.</p>`],
];

getJSON("data/analysis.json").catch(() => null).then((A) => {
  const live = A ? `<p class="dek">Right now: ${A.events.toLocaleString()} events from ${Object.keys(A.outlets).length} newsrooms, models refitted ${ago(A.generated_at)}.</p>` : "";
  $("inside").innerHTML = `
    <div class="story-head"><div class="kicker k-blue">Methods</div>
      <h2 class="hed big-hed">How every number here is made</h2>
      <p class="dek">The data, the method, how each model is checked, and what it cannot tell you. Source code for all of it is <a href="https://github.com/HalcyonVector/Nazariya" target="_blank" rel="noopener">on GitHub</a>.</p>${live}
      <nav class="toc">${M.map(([id, t]) => `<a href="#m-${id}">${esc(t)}</a>`).join("")}</nav></div>
    <div class="methods">${M.map(([id, t, q, body]) => `<section id="m-${id}"><h3 class="sub">${esc(t)}</h3>${q ? `<p class="dek">${esc(q)}</p>` : ""}${body}</section>`).join("")}</div>`;
  if (location.hash) document.querySelector(location.hash)?.scrollIntoView();
});
