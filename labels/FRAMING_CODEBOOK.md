# Framing and stance codebook (v1)

Used by the LLM labellers (silver labels), the fine-tuned classifiers, and the human checks at the
end of the build. Frames follow the Media Frames Corpus taxonomy (Boydstun et al., 2014), with
notes on how each applies to Indian news.

## 1. Frames

A frame is the angle from which an article presents its subject: which aspects it treats as the
point of the story. Pick one **primary** frame (the main angle) and up to two **secondary** frames.
Judge the article as a whole, mostly the headline and the first paragraphs.

| Frame | The article mainly treats the subject as a matter of… | Indian examples |
|---|---|---|
| `economic` | costs, prices, jobs, growth, markets, money | GST rates, fuel prices, a trade deal's tariffs, stock market reaction |
| `capacity_resources` | availability or lack of physical or human resources | doctor shortages, power/water supply, infrastructure gaps, court vacancies |
| `morality` | right and wrong, religious or ethical duty | appeals to dharma, "shameful act", religious obligation |
| `fairness_equality` | equal treatment, discrimination, rights of groups | caste discrimination, reservations, gender pay gap, minority rights |
| `legality` | laws, courts, constitutional rights, legal procedure | Supreme Court hearings, PILs, bail, constitutional validity |
| `policy` | whether a policy or scheme works or should be changed | evaluation of a government scheme, proposals, implementation gaps |
| `crime_punishment` | crimes, policing, arrests, sentences | murders, FIRs, police action, encounter killings |
| `security_defense` | threats to the nation or public, military, terrorism | border clashes, terror attacks, internal security, Naxal operations |
| `health_safety` | health, disease, accidents, disasters, safety of people | stampedes, road crashes, dengue outbreaks, food safety |
| `quality_of_life` | effects on daily life, well-being, living conditions | traffic, pollution's effect on residents, housing, civic amenities |
| `cultural_identity` | traditions, religion, caste, language, regional identity | festivals, temple/mosque disputes, language rows, Hindutva vs secularism debates |
| `public_opinion` | what the public or groups think: protests, polls, social media | protests, surveys, viral outrage, public mood |
| `political` | party politics, elections, political strategy, power | campaign statements, alliances, ministers attacking opponents, defections |
| `external_reputation` | India's relations with or standing among other countries | foreign policy, diplomatic rows, how the world sees India |
| `other` | none of the above | |

Rules of thumb:
- **The same event can be framed differently.** That is the point. A stampede can be
  `health_safety` (casualties, crowd safety), `political` (opposition blames the state government),
  or `capacity_resources` (not enough police and medical staff).
- **Politicians speaking does not make an article `political`.** It's `political` when the angle is
  party competition, elections or power, rather than the policy, crime or event itself.
- **Court cases are `legality`** unless the article's angle is clearly something else, such as a
  crime story that only mentions bail in passing (`crime_punishment`).

## 2. Stance towards an entity

For the most prominent people and parties in an article, how does the article **portray** each one?

| Stance | Meaning |
|---|---|
| `favourable` | the reader comes away with a more positive view: praise, achievements foregrounded, criticism absent or rebutted |
| `neutral` | factual, balanced or purely procedural mention |
| `unfavourable` | the reader comes away with a more negative view: criticism, failures or allegations foregrounded, defence absent or minimised |
| `unclear` | too little about the entity to tell |

For each judgement, also record its **basis**:
- `wording`: the article's own voice (headline, adjectives, what it chooses to report). This is the
  outlet's stance in the strongest sense.
- `quotes`: the stance comes only from people quoted in the article (for example, an opposition
  leader's attack reported without endorsement). This shows the outlet's *selection*, not its voice.
- `events`: the news itself is good or bad for the entity (for example, an arrest), reported plainly.

Rules of thumb:
- **Positive events are not favourable stances.** "Modi inaugurates bridge", reported plainly, is
  `neutral` with basis `events`.
- **Quote balance matters.** An attack quoted with the target's response is closer to `neutral` than
  the attack alone.
- **Headlines carry extra weight.** A loaded headline over a neutral body is still loaded.

## 3. Evidence

Each label comes with a short **paraphrase** (≤ 15 words) of what in the article supports it. Never
copy the article's sentences: the paraphrase is what gets stored and may be shown.
