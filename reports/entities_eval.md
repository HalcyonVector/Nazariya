# Entity linking evaluation

Generated 2026-10-19 20:54 UTC by `python -m nazariya.eval.entities` · judge: `local` (ollama/qwen3.5:9b-q4_K_M) · 150 articles · linker version `693ed71411`.

> **PROVISIONAL.** An LLM judges the links and lists misses; LLM judges make mistakes of their own. To be re-checked in the human labelling session.

| | precision | recall vs. alias table | table coverage | links judged | missed (in table) | not in table |
|---|---|---|---|---|---|---|
| English | 0.848 | 1.000 | 0.553 | 165 | 0 | 113 |
| Hindi | 0.815 | 0.930 | 0.477 | 81 | 5 | 78 |
| All | 0.837 | 0.976 | 0.525 | 246 | 5 | 191 |

- **precision**: share of the linker's links the judge confirms.
- **recall vs. alias table**: of correctly linkable mentions (confirmed links + names the judge found that the table could have linked), the share the linker caught.
- **table coverage**: share of judge-listed political names the table contains at all.

## Examples: links the judge rejected
- Ajay Tamta ← “Uttarakhand CM announces ₹ 96 crore worth of development projects in C”
- Selja Kumari ← “Days after quitting BJP, noted singer Jubin Nautiyal’s father Ramshara”
- Rajkumari Ratna Singh ← “CJP alleges delay in release of leaders, four sureties rejected”
- Balasahebanchi Shiv Sena ← “UP: Dalit man assaulted for staying, worshipping at Ballia temple; two”
- National Capital Territory of Delhi ← “BRS leader Harish Rao demands withdrawal of revised Godavari-Cauvery l”
- Abhijeet Dipke ← “AAP holds Tricolour march in Delhi over detention of youth at Jantar M”
- Piyush Goyal ← “US very close to finalising India trade pact: Marco Rubio before Jaish”
- Pakistan ← “Paris Curfew, Protest Ban Announced As France Seeks To Restore Order”
- Indian National Congress ← “Activists, families vulnerable to retaliation, says Sachin Pilot; pays”
- Pargat Singh ← “Activists, families vulnerable to retaliation, says Sachin Pilot; pays”
- Reliance Industries Limited ← “Activist flags pothole-riddled road stretch in Ludhiana, seeks repair”
- National Capital Territory of Delhi ← “‘Can A Plumber Land An Aircraft?’: Flydubai Indian Pilot Questions Isr”
- Rajkumari Ratna Singh ← “CJP Activists Saurav Das, Ashutosh Ranka to Be Produced Before SEM at ”
- Donald Trump ← “‘Unfortunate’: Zelenskyy rallies allies after Trump pauses Russia sanc”
- Pargat Singh ← “Punjab Congress alleges gangster-police nexus during visits to victims”
- Cockroach Janta Party ← “Gateway to growth | India Today Conclave Mumbai 2026”
- Indian National Congress ← “Virbhadra Singh’s legacy continues to inspire all Himachalis: Agnihotr”
- Supreme Court of India ← “Karnataka: Former MLC Ramesh Babu seeks action against DGP Alok”
- National Capital Territory of Delhi ← “Asian Para Games: PM Modi conveys best wishes to Indian contingent”
- National Capital Territory of Delhi ← “‘Protocol, not power’: Ex-IMF director rebuts Pakistan’s 'historic mil”
- Mallikarjun Kharge ← “One person with alleged terror links arrested, says K'taka Home Minist”
- Volodymyr Zelenskyy ← “"No one at White House ordered Starlink slowdown in Ukraine": Marco Ru”
- Mirwaiz Umar Farooq ← “Kashmir leaders condemn attack on J-K-bound bus in Haryana, seek stric”
- Priyanka Vadra ← “NSUI’s Chhotu protest as Rahul-Priyanka turn spotlight on Neha Bora, a”
- Hiteswar Saikia ← “NSUI scores successive student poll wins in Assam, bags 10 posts at B.”

## Examples: in the table but not linked
- Aam Aadmi Party ← “'साफ है कि एंटी-इंडिया गैंग कौन है', किरेन रिजिजू ने यूं की असदुद्दीन ”
- Russia ← “'बहुत जल्द...', भारत-US ट्रेड डील पर मार्को रुबियो का बड़ा बयान, जयशंक”
- M. K. Stalin ← “'मेरे नेता शेर हैं और तुम...', CM विजय को कहे अपशब्द, DMK नेता पर केस ”
- Nitin Gadkari ← “छात्रसंघ चुनाव में राहुल गांधी और अभिजीत दिपके बेअसर, ओडिशा की 8 यूनिव”
- Madhya Pradesh ← “रीवा में स्कूल बस और बोलेरो की टक्कर, 9 घायल: एक की हालत गंभीर, मैहर स”

## Most frequent names not in the table
- FSSAI (3)
- Tiger Shroff (3)
- Salman Khan (3)
- France (2)
- Food Safety and Standards Authority of India (2)
- Laurent Nunez (2)
- The Indian Express (2)
- Marco Rubio (2)
- National Students' Union of India (2)
- Lawrence Bishnoi (2)
- United Nations (2)
- Marine Le Pen (1)
- IFOP (1)
- Marianne (1)
- Algeria (1)
- Morocco (1)
- Cameroon (1)
- Federal Reserve (1)
- Union minister of state (1)
- Paris Police (1)
- Manish Malhotra (1)
- Ayushmann Khurrana (1)
- Sara Ali Khan (1)
- Ramsharan Nautiyal (1)
- Jubin Nautiyal (1)
- Ganesh Godiyal (1)
- Yashpal Arya (1)
- Sonam Wangchuk (1)
- Union Ministry of Home Affairs (1)
- MHA (1)
