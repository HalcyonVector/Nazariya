# Event-pair labelling guidelines (v1)

Each item shows two articles from different outlets, published within 72 hours of each other.
Judge what each article is **mainly about**, which is usually the headline and first paragraph,
not every topic it mentions.

| Key | Label | Use when |
|---|---|---|
| `1` | **same event** | Both mainly report the same specific happening: the same occurrence, involving the same actors, at about the same time. Different angles, details, framing or language don't matter. |
| `2` | **same story** | Both belong to the same ongoing issue or saga but mainly report **different** happenings: different days' developments, a reaction that is itself new news, a follow-up. |
| `3` | **unrelated** | Different issues, even if they share a person, place or broad theme ("both about BJP", "both about Delhi"). |
| `0` | **unsure / skip** | Not enough text to judge, an extraction error, or a genuinely ambiguous pair. Use sparingly. |

`Z` or `Backspace` undoes the last label.

## Rules of thumb
- **Same event, different framing → 1.** "SC slams ECI over voter deletions" and "SC seeks ECI
  reply on SIR" about the same hearing are the same event. Framing differences are exactly what
  the project measures later.
- **New statement = new event.** If a politician's reaction to yesterday's verdict is the main
  news, it's a new event in the same story → `2`. If the reaction is only a quote inside a report
  of the verdict, it's still the verdict → `1`.
- **Same incident, different stage → 2.** An accident, then the arrest of the driver two days later.
- **Roundups and live blogs:** label by their main subject. If there isn't one, use `0`.
- **Language doesn't matter.** An English and a Hindi report of the same happening are `1`.
- **Explainers and opinion pieces** about a happening are `1` only if they are clearly about that
  specific happening; general commentary on the issue is `2`.
- **Don't open the links** unless the excerpt is unclear. Most pairs should take under 10 seconds.

About 20 pairs reappear later with their sides swapped. That's deliberate: it measures how
consistent the labels are, so label each one fresh.
