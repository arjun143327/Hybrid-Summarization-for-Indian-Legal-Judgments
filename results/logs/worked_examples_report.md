# Phase 3 Supplement: Worked Examples, Chance Baseline, k_i Audit

Date: 2026-09-13 18:46:48

## 1. k_i Budget Rule — Confirmed

**Formula**: `k_i = max(3, round(0.05 * M_i))`

- **Rounding mode**: Python built-in `round()` which uses *banker's rounding* (round-half-to-even). Concretely: `round(0.5)=0`, `round(1.5)=2`, `round(2.5)=2`, `round(3.5)=4`.
- **Symmetry**: the SAME `k_i` value (computed once from `M_i`) is used for BOTH the true-label top-k extraction and the predicted-score top-k extraction. The two sides of the Jaccard / recall comparison always have equal set size. **CONFIRMED.**
- **Short-document tail** (M_i < 20): `0.05 * M_i < 1.0`. After `round()` the result is 0 or 1. `max(3, ...)` clamps to `k_i = 3`. The floor clause handles ALL short documents.

Short-document examples:

| M_i | 0.05 * M_i | after round() | k_i (final) | Note |
|:---:|:---:|:---:|:---:|---|
| 1 | 0.05 | 0 | 3 | floor fires |
| 5 | 0.25 | 0 | 3 | floor fires |
| 10 | 0.50 | 0 | 3 | round(0.5)=0 (banker), floor fires |
| 15 | 0.75 | 1 | 3 | floor fires |
| 19 | 0.95 | 1 | 3 | floor fires |
| 20 | 1.00 | 1 | 3 | floor fires |
| 21 | 1.05 | 1 | 3 | floor fires |
| 60 | 3.00 | 3 | 3 | round(3.0)=3, max(3,3)=3 — borderline |
| 61 | 3.05 | 3 | 3 | max(3,3)=3 |
| 80 | 4.00 | 4 | 4 | floor inactive |

**k_i corpus statistics (1,054 val docs)**:

| Statistic | Value |
|---|:---:|
| Docs where `floor=3` clause fires | `139` |
| Docs with M_i < 20 | `4` |
| Min M_i in val split | `14` → k_i = `3` |
| Max M_i in val split | `1189` → k_i = `59` |
| Average k_i | `7.15` |
| Median k_i | `5` |

## 2. Real vs. Chance Baseline

> **Chance method**: for each of the 1,054 val documents, 200 independent
> random top-k_i selections (uniform, without replacement) from M_i sentences.
> Jaccard and recall computed against a fixed oracle top-k_i set of size k_i.
> Results averaged over all docs with the same aggregation as the real metrics.
> Seed: 777, per-doc seeds drawn from this root RNG.

| Metric | C1 (Replica) | C2 (Redundancy Swap) | C3 (Proposed) | **CHANCE** |
|---|:---:|:---:|:---:|:---:|
| **Top-k Jaccard (mean)** | `0.0735` | `0.0735` | `0.0640` | **`0.0311`** |
| **Top-k % Recall (mean)** | `0.1225` | `0.1225` | `0.1068` | **`0.0543`** |

- C1/C2 Jaccard (`0.0735`) is **2.36x** above chance (`0.0311`).
- C3 Jaccard (`0.0640`) is **2.06x** above chance (`0.0311`).
- All three configs are meaningfully above random sentence selection.

> **Note on absolute values**: the chance Jaccard is also low because k_i is only ~5% of M_i. At k/M = 0.05 and M~144, random selection yields |intersection| ~ k^2/M ~ 0.003 per doc, giving Jaccard ~ k^2/(2Mk - k^2) ~ 0.025. The simulated value confirms this regime.

## 3. Worked Examples — Configs C1 and C3

> **True top-k**: ranked by ground-truth ROUGE label `y_ij` (descending).
> **Pred top-k**: ranked by GBR predicted score `y_hat_ij` (descending).
> **MATCH** = sentence appears in BOTH sets. **miss** = appears in one only.

---

### SHORT DOCUMENT — doc_id=`4820`, M_i=14, k_i=3

**k_i derivation**: `max(3, round(0.05 * 14)) = max(3, round(0.7000)) = max(3, 1) = 3`

#### Config C1 — Jaccard=`0.5`, Recall=`67%` (2/3 matched)

**True top-3** (ranked by y_ij, descending):

| Rank | Sent idx | y_ij | In Pred? | Text snippet |
|:---:|:---:|:---:|:---:|---|
| 1 | 3 | `0.5538` | MATCH | In the first paragraph of the petition it is stated "This is an application for Review of the order dated 9.12... |
| 2 | 1 | `0.5161` | miss | In Special Leave Petition (C) No. 13618 of 1983 CHAMBER MATTER By Circulation The order of the Court was deliv... |
| 3 | 11 | `0.4884` | MATCH | We must however express our deep dissatisfaction and anguish with the indiscriminate manner in which petitions... |

**Predicted top-3** (ranked by y_hat, descending):

| Rank | Sent idx | y_hat | True y | In Oracle? | Text snippet |
|:---:|:---:|:---:|:---:|:---:|---|
| 1 | 11 | `0.5871` | `0.4884` | MATCH | We must however express our deep dissatisfaction and anguish with the indiscriminate manner in which petitions... |
| 2 | 3 | `0.5768` | `0.5538` | MATCH | In the first paragraph of the petition it is stated "This is an application for Review of the order dated 9.12... |
| 3 | 6 | `0.5687` | `0.3738` | miss | In the second paragraph we are told that no detailed grounds have been taken (though in point of fact not a si... |

#### Config C3 — Jaccard=`0.5`, Recall=`67%` (2/3 matched)

**True top-3** (ranked by y_ij, descending):

| Rank | Sent idx | y_ij | In Pred? | Text snippet |
|:---:|:---:|:---:|:---:|---|
| 1 | 3 | `0.5538` | MATCH | In the first paragraph of the petition it is stated "This is an application for Review of the order dated 9.12... |
| 2 | 1 | `0.5161` | miss | In Special Leave Petition (C) No. 13618 of 1983 CHAMBER MATTER By Circulation The order of the Court was deliv... |
| 3 | 11 | `0.4884` | MATCH | We must however express our deep dissatisfaction and anguish with the indiscriminate manner in which petitions... |

**Predicted top-3** (ranked by y_hat, descending):

| Rank | Sent idx | y_hat | True y | In Oracle? | Text snippet |
|:---:|:---:|:---:|:---:|:---:|---|
| 1 | 3 | `0.5316` | `0.5538` | MATCH | In the first paragraph of the petition it is stated "This is an application for Review of the order dated 9.12... |
| 2 | 11 | `0.5049` | `0.4884` | MATCH | We must however express our deep dissatisfaction and anguish with the indiscriminate manner in which petitions... |
| 3 | 4 | `0.4937` | `0.4` | miss | The said order discloses an error apparent on the face of the record as will be clear from perusal of the vari... |

---

### AVERAGE DOCUMENT — doc_id=`1005`, M_i=83, k_i=4

**k_i derivation**: `max(3, round(0.05 * 83)) = max(3, round(4.1500)) = max(3, 4) = 4`

#### Config C1 — Jaccard=`0.0`, Recall=`0%` (0/4 matched)

**True top-4** (ranked by y_ij, descending):

| Rank | Sent idx | y_ij | In Pred? | Text snippet |
|:---:|:---:|:---:|:---:|---|
| 1 | 59 | `0.8627` | miss | There is nothing in the provisions of article 366(22) which requires a court to recognise such a person as a R... |
| 2 | 61 | `0.7945` | miss | As the appellant was an 'ex Ruler ', he was within the class of persons who were by name specifically included... |
| 3 | 66 | `0.7018` | miss | A maufidar could be a person who was the holder of land which was exempted from the payment of rent or tax. |
| 4 | 10 | `0.6667` | miss | The appellant was the Ruler of the State of Baster. |

**Predicted top-4** (ranked by y_hat, descending):

| Rank | Sent idx | y_hat | True y | In Oracle? | Text snippet |
|:---:|:---:|:---:|:---:|:---:|---|
| 1 | 79 | `0.5854` | `0.4308` | miss | Since it was a question of fact whether the villages had been assessed to land revenue, which was denied on be... |
| 2 | 70 | `0.5506` | `0.3371` | miss | In the petition under articles 226 and 227 of the Constitution, filed by the appellant in the High Court, it w... |
| 3 | 80 | `0.5468` | `0.4124` | miss | As for the other villages, in Schedules A and B of the petition of the appellant under articles 226 and 227 of... |
| 4 | 68 | `0.5394` | `0.3689` | miss | It is, however, contended on behalf of the appellant that the most important part of the definition was the co... |

#### Config C3 — Jaccard=`0.0`, Recall=`0%` (0/4 matched)

**True top-4** (ranked by y_ij, descending):

| Rank | Sent idx | y_ij | In Pred? | Text snippet |
|:---:|:---:|:---:|:---:|---|
| 1 | 59 | `0.8627` | miss | There is nothing in the provisions of article 366(22) which requires a court to recognise such a person as a R... |
| 2 | 61 | `0.7945` | miss | As the appellant was an 'ex Ruler ', he was within the class of persons who were by name specifically included... |
| 3 | 66 | `0.7018` | miss | A maufidar could be a person who was the holder of land which was exempted from the payment of rent or tax. |
| 4 | 10 | `0.6667` | miss | The appellant was the Ruler of the State of Baster. |

**Predicted top-4** (ranked by y_hat, descending):

| Rank | Sent idx | y_hat | True y | In Oracle? | Text snippet |
|:---:|:---:|:---:|:---:|:---:|---|
| 1 | 79 | `0.5452` | `0.4308` | miss | Since it was a question of fact whether the villages had been assessed to land revenue, which was denied on be... |
| 2 | 69 | `0.5167` | `0.3059` | miss | It was contended that even if the appellant was a maufidar, there was nothing to show that with reference to a... |
| 3 | 74 | `0.5164` | `0.3704` | miss | There is thus no material on the record to establish that the appellant as a maufidar had no right to recover ... |
| 4 | 70 | `0.513` | `0.3371` | miss | In the petition under articles 226 and 227 of the Constitution, filed by the appellant in the High Court, it w... |

---

### LONG DOCUMENT — doc_id=`6778`, M_i=1189, k_i=59

**k_i derivation**: `max(3, round(0.05 * 1189)) = max(3, round(59.4500)) = max(3, 59) = 59`

#### Config C1 — Jaccard=`0.0172`, Recall=`3%` (2/59 matched)

**True top-59** (ranked by y_ij, descending):

| Rank | Sent idx | y_ij | In Pred? | Text snippet |
|:---:|:---:|:---:|:---:|---|
| 1 | 262 | `1.0` | miss | Clause (c) states that in the case of any other person the sanction would be of the authority competent to rem... |
| 2 | 519 | `1.0` | miss | Any complaint against a Judge and its investigation by the CBI, if given publicity will have a far reaching im... |
| 3 | 518 | `1.0` | miss | But these safeguards may not be adequate. |
| 4 | 522 | `1.0` | miss | They should be protected. |
| 5 | 651 | `1.0` | miss | The Investigating Officer is only required to collect material to find out whether the offence alleged appears... |
| 6 | 530 | `1.0` | miss | All that is required is to lay down certain guidelines lest the Act may be misused. |
| 7 | 277 | `1.0` | miss | Indeed, they are not Judges if they allow themselves to be guided by the Government in the performance of thei... |
| 8 | 535 | `1.0` | miss | Indeed, the court 's role today is much more. |
| 9 | 536 | `1.0` | miss | It is expanding beyond dispute settling and interstitial law making. |
| 10 | 284 | `1.0` | miss | The 'general word is presumed to be restricted to the same genus as those of the particular and specific words... |
| 11 | 1181 | `1.0` | miss | If this was the expectation of the framers of the Constitution and their vision of the moral fibre in the high... |
| 12 | 1052 | `1.0` | miss | Such a law in the form of the and the rules framed thereunder has been enacted. |
| 13 | 415 | `1.0` | miss | It will not have the consequence of removing the Judge from the office unless it is followed by an order of th... |
| 14 | 1056 | `1.0` | miss | The need for these special provisions is a clear pointer in the direction of inapplicability to them of the ge... |
| 15 | 681 | `1.0` | miss | The standards of judicial behaviour, both on and off the Bench, are normally extremely high. |
| 16 | 682 | `1.0` | miss | For a Judge to deviate from such standards of honesty and impar tiality is to betray the trust reposed on him. |
| 17 | 427 | `1.0` | miss | The order of the President is sine qua non for removal of a Judge. |
| 18 | 428 | `1.0` | miss | The President alone could make that order. |
| 19 | 684 | `1.0` | miss | From the standpoint of justice the size of the bribe or scope of corruption cannot be the scale for measuring ... |
| 20 | 683 | `1.0` | miss | No excuse or no legal relativity can condone such betrayal. |
| 21 | 687 | `1.0` | miss | The slightest hint of irregu larity or impropriety in the Court is a cause for great anxiety and alarm. |
| 22 | 447 | `1.0` | miss | The President is not an outsider so far judiciary is concerned. |
| 23 | 1089 | `1.0` | miss | The Constitution, while providing that their position would be akin to that of a Judge of the Supreme Court, c... |
| 24 | 324 | `1.0` | miss | It is a general term and general term in the Act should not be narrowly construed. |
| 25 | 326 | `1.0` | miss | There is no such indication to the contrary in the Act. |
| 26 | 1102 | `1.0` | miss | The clear legislative intent is that the enactment applies only to those in whose case sanction of this kind i... |
| 27 | 336 | `1.0` | miss | The Executive is competent to appoint the Judges but not empowered to remove them. |
| 28 | 732 | `1.0` | miss | So far this aspect is concerned, the two categories of Judges High Court and Supreme Court Judges on the one h... |
| 29 | 477 | `1.0` | miss | One is the power of Parliament and the other is the jurisdiction of a Criminal Court. |
| 30 | 478 | `1.0` | miss | Both are mutually exclusive. |
| 31 | 740 | `1.0` | miss | He cannot be convicted and pun ished. |
| 32 | 625 | `1.0` | miss | (Section 106 of the Evidence Act). |
| 33 | 627 | `1.0` | miss | It is for him to explain. |
| 34 | 247 | `1.0` | miss | The competent authority may refuse sanction for prosecution if the offence alleged has no material to support ... |
| 35 | 764 | `1.0` | miss | This will seriously jeopardise the independence of judiciary which is undoubtedly a basic feature of the Const... |
| 36 | 710 | `0.9863` | miss | Thus although more than one person are involved in the process, it is not permissible to say that no authority... |
| 37 | 1110 | `0.9841` | miss | In fact, the very need to read the proposed guidelines in the exist ing law by implication is a clear indicati... |
| 38 | 454 | `0.9778` | miss | Parliament has no part to play in the matter of appointment of Judges except that the Executive is responsible... |
| 39 | 482 | `0.9737` | miss | It is not objectionable to initiate criminal proceedings against public servant before exhausting the discipli... |
| 40 | 1108 | `0.973` | miss | The remedy is not to extend the existing law and make it workable by reading into it certain guidelines for wh... |
| 41 | 550 | `0.9697` | miss | If the Chief Justice of India himself is the person against whom the allegations of criminal misconduct are re... |
| 42 | 1017 | `0.967` | miss | Where no such relation ship exists in the absence of any vertical hierarchy and the holder of the public offic... |
| 43 | 1111 | `0.9667` | miss | Making the law applicable with the aid of the suggested guidelines, is not in the domain of judicial craftmans... |
| 44 | 408 | `0.9663` | miss | If the literal meaning of the legislative language used would lead to results which would defeat the purpose o... |
| 45 | 1049 | `0.963` | miss | The special law envisaged by Article 124(5) for dealing with the misbehaviour of a Judge covers the field of '... |
| 46 | 631 | `0.963` | miss | The principle is applied in the absence of statutory provision to the contrary. |
| 47 | 655 | `0.96` | miss | He should be taken into confidence if he is willing to cooperate. |
| 48 | 731 | `0.96` | miss | Similarly protection is available to the High Court and Supreme Court Judges through the provisions of Article... |
| 49 | 1180 | `0.9577` | MATCH | It appears that for a rare aberrant at that level, unless he resigned when faced with such a situation, remova... |
| 50 | 1122 | `0.9565` | miss | The fact that the Parliament did not enact any other law even then for the investigation into allegations of c... |
| 51 | 551 | `0.9451` | miss | There shall be similar consul tation at the stage of examining the question of granting sanction for prosecuti... |
| 52 | 1051 | `0.9429` | miss | Thus, even for the procedure for investigation into any misbehaviour of a Judge as well as its proof, a law en... |
| 53 | 413 | `0.9429` | miss | The expression "the authority competent to remove" used in clause (c) of Section 6(1) is to be construed to me... |
| 54 | 763 | `0.9429` | miss | If the President is held to be the appropriate authority to grant the sanction without reference to the Parlia... |
| 55 | 686 | `0.9412` | miss | A judicial scandal has always been regarded as far more deplorable than a scandal involving either the Executi... |
| 56 | 1081 | `0.9412` | miss | If the Act is applicable to Judges of the High Courts and the Supreme Court, it is obvious that the same must ... |
| 57 | 1096 | `0.9412` | MATCH | The ambit of the enactment is to be determined on the basis of the public office held by the public serv ant, ... |
| 58 | 849 | `0.9412` | miss | There is no material to indicate that corruption in judiciary was a mischief to be cured when the Prevention o... |
| 59 | 753 | `0.9412` | miss | If the legislature had intended to exclude the High Court and Supreme Court Judges from the field of Section 5... |

**Predicted top-59** (ranked by y_hat, descending):

| Rank | Sent idx | y_hat | True y | In Oracle? | Text snippet |
|:---:|:---:|:---:|:---:|:---:|---|
| 1 | 1099 | `0.5367` | `0.6931` | miss | It is for the purpose of construing the provisions of the enactment and determining the scope and ambit thereo... |
| 2 | 15 | `0.5304` | `0.4844` | miss | Section 6(1)(c) specifically enjoins that no court shall take cognizance of an offence punishable under Sectio... |
| 3 | 1184 | `0.529` | `0.9219` | miss | If it is considered that the situation has altered requiring scrutiny of the conduct of even Judges at the hig... |
| 4 | 952 | `0.525` | `0.5191` | miss | In other words, the argument is that not withstanding the fact that no sanction was required for prosecution o... |
| 5 | 981 | `0.5216` | `0.4483` | miss | And this interrelation clearly provides a clue to the understanding of the provision in Section 6 providing fo... |
| 6 | 951 | `0.5197` | `0.5175` | miss | It was, however, contended that for the purpose of deciding the question of applicability of the Act to the ap... |
| 7 | 1113 | `0.519` | `0.4667` | miss | A charge of corruption against a superior Judge amounting to criminal misconduct by abuse of his office would ... |
| 8 | 1185 | `0.5175` | `0.9204` | miss | Any attempt to bring the Judges of the High Courts and the Supreme Court within the purview of the Prevention ... |
| 9 | 1076 | `0.5123` | `0.9171` | miss | These words in clause (e) have to be given some meaning which would place the burden on the prosecution, howso... |
| 10 | 1092 | `0.5098` | `0.4425` | miss | It was also suggested at the hearing that the absence of need of sanction for prosecution under Section 6 of t... |
| 11 | 1078 | `0.5074` | `0.8873` | miss | In the case of such public servants whenever sanction to prosecute is sought under Section 6 of the Act, the c... |
| 12 | 1057 | `0.5073` | `0.8485` | miss | Construction of Section 6(1)(c) of the Act as suggested by the learned Solicitor General by treating the Presi... |
| 13 | 1132 | `0.5071` | `0.48` | miss | The collec tive wisdom of the constitutional functionaries involved in the process of appointing a superior Ju... |
| 14 | 1070 | `0.5071` | `0.8493` | miss | The fact remains that while according sanction to prosecute under Section 6 of the Act, the competent authorit... |
| 15 | 16 | `0.5069` | `0.8211` | miss | So to initiate a proceeding against a Judge of a Supreme Court for criminal misconduct failing under Section 5... |
| 16 | 52 | `0.5048` | `0.5376` | miss | The third most crucial question that fails for consider ation in this case is who is the competent authority t... |
| 17 | 1137 | `0.5045` | `0.3415` | miss | It is, therefore, time that all the constitutional functionaries involved in the process of appointment of sup... |
| 18 | 1025 | `0.5044` | `0.3158` | miss | "Therefore, the important question that arises in such cases of suspect ed misbehaviour and/or reported lack o... |
| 19 | 1176 | `0.5041` | `0.3613` | miss | With this duty entrusted to the higher judiciary, it was natural to expect that the higher judiciary would not... |
| 20 | 73 | `0.5039` | `0.5301` | miss | In these circumstances the only question to be consid ered is who will be the authority or who is the authorit... |
| 21 | 1059 | `0.5036` | `0.7105` | miss | This is more so, since the rejection of such an argument would not in any manner jeopardise the provisions of ... |
| 22 | 954 | `0.5026` | `0.5287` | miss | The question of grant of sanction under Section 6 for the prosecution of a Judge or Chief Justice of a High Co... |
| 23 | 955 | `0.5026` | `0.5484` | miss | Clauses (a), (b) and (c) in Sub section (1) of Section 6 exhaus 286 tively provide for the competent authority... |
| 24 | 1125 | `0.5021` | `0.4167` | miss | Maybe, need is now felt for a law providing for trial and punishment of a superior Judge who is charged with t... |
| 25 | 987 | `0.5017` | `0.4` | miss | That authority alone would be competent to judge whether on the facts alleged, there has been an abuse or misu... |
| 26 | 761 | `0.5007` | `0.427` | miss | Before proceeding further 1 would again state that having answered the question as to whether a Judge of the s... |
| 27 | 984 | `0.4999` | `0.3855` | miss | A grant of sanction is not an idle formality but a solemn and sacrosanct act which removes the umbrella of pro... |
| 28 | 995 | `0.4996` | `0.4969` | miss | It would follow that where the office held by the public servant is not a part of a verti cal hierarchy in whi... |
| 29 | 1144 | `0.4995` | `0.3956` | miss | It was expected that the superior Judges who were constituted into a different class and created as superior m... |
| 30 | 1065 | `0.499` | `0.7529` | miss | These are strong reasons to hold that Section 6(1)(c) of the Act is inappliable to a Judge of a High Court or ... |
| 31 | 1066 | `0.4985` | `0.8052` | miss | An additional reason 'indicating inapplicability of the Act is the practical difficulty in applying criminal m... |
| 32 | 1129 | `0.4985` | `0.4167` | miss | The framers of the Constitution had visualised that the constitutional scheme for appointment of the superior ... |
| 33 | 1023 | `0.4982` | `0.3881` | miss | In my view since the question relates to the continuance of a high constitutional functionary like the Additio... |
| 34 | 996 | `0.497` | `0.825` | miss | The decisions of this Court have unequivocally held that a Judge or Chief Justice of a High Court is a constit... |
| 35 | 1094 | `0.4964` | `0.9195` | miss | The need for sanction under Section 6 for prosecution of the holder of a public office indicates the ambit and... |
| 36 | 1048 | `0.4953` | `0.8632` | miss | There can be no doubt that the expression 'misbehaviour ' is of wide import and includes within its ambit crim... |
| 37 | 838 | `0.4941` | `0.4615` | miss | It was argued that in this manner preservation of independence of the judiciary could be ensured while treat i... |
| 38 | 1151 | `0.4933` | `0.9306` | miss | The view that Judges of the High Courts and the Supreme Court are outside the purview of the Prevention of Cor... |
| 39 | 700 | `0.4933` | `0.5` | miss | For the purpose of this argument it is presumed that there is no authority competent to remove a High Court Ju... |
| 40 | 1096 | `0.4931` | `0.9412` | MATCH | The ambit of the enactment is to be determined on the basis of the public office held by the public serv ant, ... |
| 41 | 14 | `0.4915` | `0.7387` | miss | Therefore, it is clear that a Judge will be liable for committing criminal misconduct within the meaning of cl... |
| 42 | 71 | `0.4903` | `0.8571` | miss | Therefore a Judge of the High Court or of the Supreme Court comes within the defini tion of public servant and... |
| 43 | 1010 | `0.4896` | `0.44` | miss | Since the order of removal in such a case is to be made by the President, the learned Solicitor General conten... |
| 44 | 1118 | `0.4893` | `0.4754` | miss | It appears that the framers of the Constitution did not contem plate the need for prosecution of a Judge at th... |
| 45 | 982 | `0.489` | `0.5051` | miss | Therefore, it unquestionably follows that the sanction to prosecute can be given by an authority competent to ... |
| 46 | 1080 | `0.4877` | `0.75` | miss | It does appear that this too is a pointer in the direction that even after the 1964 amendment of the Act follo... |
| 47 | 1034 | `0.4875` | `0.4255` | miss | As the law now stands it is not open to any single individual, whether it is the President or the Chief Justic... |
| 48 | 1013 | `0.4872` | `0.8923` | miss | Section 6(1)(c) speaks of 'authority competent to remove ' which plainly indicates the substantive competence ... |
| 49 | 68 | `0.487` | `0.5556` | miss | It has been, therefore, urged that Section 6(i)(C) of the Prevention of Corruption Act, 1947 is not applicable... |
| 50 | 1060 | `0.4869` | `0.4146` | miss | It can also not be overlooked that the Santhanam Commit tee Report did not consider the judiciary within its p... |
| 51 | 1042 | `0.4863` | `0.4304` | miss | The scheme of the exist ing law to deal with such situations was considered at length and it was also held tha... |
| 52 | 994 | `0.4861` | `0.4242` | miss | In other words, Section 6 applies only in cases where there is a vertical hierarchy of public offices and the ... |
| 53 | 103 | `0.4858` | `0.7363` | miss | The President, therefore, being the authority competent to appoint and to remove a Judge, of course in accorda... |
| 54 | 1097 | `0.4854` | `0.9402` | miss | In other words, if the holder of a public office during his tenure in office cannot be prosecuted without sanc... |
| 55 | 958 | `0.484` | `0.8667` | miss | It follows that the holder of an office, even though a 'public servant ' accord ing to the definition in the A... |
| 56 | 1180 | `0.4837` | `0.9577` | MATCH | It appears that for a rare aberrant at that level, unless he resigned when faced with such a situation, remova... |
| 57 | 979 | `0.483` | `0.466` | miss | The expression 'office ' in the three sub clauses of Section 6(1) would clear ly denote that office which the ... |
| 58 | 1183 | `0.4829` | `0.931` | miss | Clearly, it was expected that the higher judiciary whose word would be final in the interpretation of all laws... |
| 59 | 975 | `0.4824` | `0.475` | miss | The offence would be committed by the public servant by misusing or abusing the power of office and it is from... |

#### Config C3 — Jaccard=`0.0442`, Recall=`8%` (5/59 matched)

**True top-59** (ranked by y_ij, descending):

| Rank | Sent idx | y_ij | In Pred? | Text snippet |
|:---:|:---:|:---:|:---:|---|
| 1 | 262 | `1.0` | miss | Clause (c) states that in the case of any other person the sanction would be of the authority competent to rem... |
| 2 | 519 | `1.0` | miss | Any complaint against a Judge and its investigation by the CBI, if given publicity will have a far reaching im... |
| 3 | 518 | `1.0` | miss | But these safeguards may not be adequate. |
| 4 | 522 | `1.0` | miss | They should be protected. |
| 5 | 651 | `1.0` | miss | The Investigating Officer is only required to collect material to find out whether the offence alleged appears... |
| 6 | 530 | `1.0` | miss | All that is required is to lay down certain guidelines lest the Act may be misused. |
| 7 | 277 | `1.0` | miss | Indeed, they are not Judges if they allow themselves to be guided by the Government in the performance of thei... |
| 8 | 535 | `1.0` | miss | Indeed, the court 's role today is much more. |
| 9 | 536 | `1.0` | miss | It is expanding beyond dispute settling and interstitial law making. |
| 10 | 284 | `1.0` | miss | The 'general word is presumed to be restricted to the same genus as those of the particular and specific words... |
| 11 | 1181 | `1.0` | miss | If this was the expectation of the framers of the Constitution and their vision of the moral fibre in the high... |
| 12 | 1052 | `1.0` | miss | Such a law in the form of the and the rules framed thereunder has been enacted. |
| 13 | 415 | `1.0` | miss | It will not have the consequence of removing the Judge from the office unless it is followed by an order of th... |
| 14 | 1056 | `1.0` | MATCH | The need for these special provisions is a clear pointer in the direction of inapplicability to them of the ge... |
| 15 | 681 | `1.0` | miss | The standards of judicial behaviour, both on and off the Bench, are normally extremely high. |
| 16 | 682 | `1.0` | miss | For a Judge to deviate from such standards of honesty and impar tiality is to betray the trust reposed on him. |
| 17 | 427 | `1.0` | miss | The order of the President is sine qua non for removal of a Judge. |
| 18 | 428 | `1.0` | miss | The President alone could make that order. |
| 19 | 684 | `1.0` | miss | From the standpoint of justice the size of the bribe or scope of corruption cannot be the scale for measuring ... |
| 20 | 683 | `1.0` | miss | No excuse or no legal relativity can condone such betrayal. |
| 21 | 687 | `1.0` | miss | The slightest hint of irregu larity or impropriety in the Court is a cause for great anxiety and alarm. |
| 22 | 447 | `1.0` | miss | The President is not an outsider so far judiciary is concerned. |
| 23 | 1089 | `1.0` | miss | The Constitution, while providing that their position would be akin to that of a Judge of the Supreme Court, c... |
| 24 | 324 | `1.0` | miss | It is a general term and general term in the Act should not be narrowly construed. |
| 25 | 326 | `1.0` | miss | There is no such indication to the contrary in the Act. |
| 26 | 1102 | `1.0` | MATCH | The clear legislative intent is that the enactment applies only to those in whose case sanction of this kind i... |
| 27 | 336 | `1.0` | miss | The Executive is competent to appoint the Judges but not empowered to remove them. |
| 28 | 732 | `1.0` | miss | So far this aspect is concerned, the two categories of Judges High Court and Supreme Court Judges on the one h... |
| 29 | 477 | `1.0` | miss | One is the power of Parliament and the other is the jurisdiction of a Criminal Court. |
| 30 | 478 | `1.0` | miss | Both are mutually exclusive. |
| 31 | 740 | `1.0` | miss | He cannot be convicted and pun ished. |
| 32 | 625 | `1.0` | miss | (Section 106 of the Evidence Act). |
| 33 | 627 | `1.0` | miss | It is for him to explain. |
| 34 | 247 | `1.0` | miss | The competent authority may refuse sanction for prosecution if the offence alleged has no material to support ... |
| 35 | 764 | `1.0` | miss | This will seriously jeopardise the independence of judiciary which is undoubtedly a basic feature of the Const... |
| 36 | 710 | `0.9863` | miss | Thus although more than one person are involved in the process, it is not permissible to say that no authority... |
| 37 | 1110 | `0.9841` | miss | In fact, the very need to read the proposed guidelines in the exist ing law by implication is a clear indicati... |
| 38 | 454 | `0.9778` | miss | Parliament has no part to play in the matter of appointment of Judges except that the Executive is responsible... |
| 39 | 482 | `0.9737` | miss | It is not objectionable to initiate criminal proceedings against public servant before exhausting the discipli... |
| 40 | 1108 | `0.973` | miss | The remedy is not to extend the existing law and make it workable by reading into it certain guidelines for wh... |
| 41 | 550 | `0.9697` | miss | If the Chief Justice of India himself is the person against whom the allegations of criminal misconduct are re... |
| 42 | 1017 | `0.967` | miss | Where no such relation ship exists in the absence of any vertical hierarchy and the holder of the public offic... |
| 43 | 1111 | `0.9667` | MATCH | Making the law applicable with the aid of the suggested guidelines, is not in the domain of judicial craftmans... |
| 44 | 408 | `0.9663` | miss | If the literal meaning of the legislative language used would lead to results which would defeat the purpose o... |
| 45 | 1049 | `0.963` | miss | The special law envisaged by Article 124(5) for dealing with the misbehaviour of a Judge covers the field of '... |
| 46 | 631 | `0.963` | miss | The principle is applied in the absence of statutory provision to the contrary. |
| 47 | 655 | `0.96` | miss | He should be taken into confidence if he is willing to cooperate. |
| 48 | 731 | `0.96` | miss | Similarly protection is available to the High Court and Supreme Court Judges through the provisions of Article... |
| 49 | 1180 | `0.9577` | miss | It appears that for a rare aberrant at that level, unless he resigned when faced with such a situation, remova... |
| 50 | 1122 | `0.9565` | miss | The fact that the Parliament did not enact any other law even then for the investigation into allegations of c... |
| 51 | 551 | `0.9451` | miss | There shall be similar consul tation at the stage of examining the question of granting sanction for prosecuti... |
| 52 | 1051 | `0.9429` | MATCH | Thus, even for the procedure for investigation into any misbehaviour of a Judge as well as its proof, a law en... |
| 53 | 413 | `0.9429` | miss | The expression "the authority competent to remove" used in clause (c) of Section 6(1) is to be construed to me... |
| 54 | 763 | `0.9429` | miss | If the President is held to be the appropriate authority to grant the sanction without reference to the Parlia... |
| 55 | 686 | `0.9412` | miss | A judicial scandal has always been regarded as far more deplorable than a scandal involving either the Executi... |
| 56 | 1081 | `0.9412` | miss | If the Act is applicable to Judges of the High Courts and the Supreme Court, it is obvious that the same must ... |
| 57 | 1096 | `0.9412` | MATCH | The ambit of the enactment is to be determined on the basis of the public office held by the public serv ant, ... |
| 58 | 849 | `0.9412` | miss | There is no material to indicate that corruption in judiciary was a mischief to be cured when the Prevention o... |
| 59 | 753 | `0.9412` | miss | If the legislature had intended to exclude the High Court and Supreme Court Judges from the field of Section 5... |

**Predicted top-59** (ranked by y_hat, descending):

| Rank | Sent idx | y_hat | True y | In Oracle? | Text snippet |
|:---:|:---:|:---:|:---:|:---:|---|
| 1 | 1184 | `0.5889` | `0.9219` | miss | If it is considered that the situation has altered requiring scrutiny of the conduct of even Judges at the hig... |
| 2 | 1076 | `0.5658` | `0.9171` | miss | These words in clause (e) have to be given some meaning which would place the burden on the prosecution, howso... |
| 3 | 1176 | `0.5545` | `0.3613` | miss | With this duty entrusted to the higher judiciary, it was natural to expect that the higher judiciary would not... |
| 4 | 981 | `0.5537` | `0.4483` | miss | And this interrelation clearly provides a clue to the understanding of the provision in Section 6 providing fo... |
| 5 | 1150 | `0.5502` | `0.4368` | miss | It is for the Parliament to decide whether that stage has reached in the superior judiciary when legal sanctio... |
| 6 | 1070 | `0.5496` | `0.8493` | miss | The fact remains that while according sanction to prosecute under Section 6 of the Act, the competent authorit... |
| 7 | 1041 | `0.5488` | `0.4082` | miss | It was also pointed out that the object served in this manner was the greater public interest to preserve inde... |
| 8 | 1099 | `0.5474` | `0.6931` | miss | It is for the purpose of construing the provisions of the enactment and determining the scope and ambit thereo... |
| 9 | 1075 | `0.5455` | `0.4354` | miss | Prescribing the substantive offence by insertion of clause (e.) as a part of the same schem 297 of amendment a... |
| 10 | 955 | `0.5433` | `0.5484` | miss | Clauses (a), (b) and (c) in Sub section (1) of Section 6 exhaus 286 tively provide for the competent authority... |
| 11 | 15 | `0.543` | `0.4844` | miss | Section 6(1)(c) specifically enjoins that no court shall take cognizance of an offence punishable under Sectio... |
| 12 | 1078 | `0.5365` | `0.8873` | miss | In the case of such public servants whenever sanction to prosecute is sought under Section 6 of the Act, the c... |
| 13 | 1025 | `0.5337` | `0.3158` | miss | "Therefore, the important question that arises in such cases of suspect ed misbehaviour and/or reported lack o... |
| 14 | 838 | `0.5332` | `0.4615` | miss | It was argued that in this manner preservation of independence of the judiciary could be ensured while treat i... |
| 15 | 951 | `0.5324` | `0.5175` | miss | It was, however, contended that for the purpose of deciding the question of applicability of the Act to the ap... |
| 16 | 1111 | `0.5318` | `0.9667` | MATCH | Making the law applicable with the aid of the suggested guidelines, is not in the domain of judicial craftmans... |
| 17 | 1096 | `0.5307` | `0.9412` | MATCH | The ambit of the enactment is to be determined on the basis of the public office held by the public serv ant, ... |
| 18 | 1132 | `0.5286` | `0.48` | miss | The collec tive wisdom of the constitutional functionaries involved in the process of appointing a superior Ju... |
| 19 | 1051 | `0.5284` | `0.9429` | MATCH | Thus, even for the procedure for investigation into any misbehaviour of a Judge as well as its proof, a law en... |
| 20 | 826 | `0.5283` | `0.5234` | miss | His first contention is that the Judges of the High Courts and the Supreme Court are not within the purview of... |
| 21 | 953 | `0.5278` | `0.5577` | miss | It is argued that if the grant of sanction under Section 6 of the Act for prosecution of the incumbent for the... |
| 22 | 1071 | `0.5277` | `0.8889` | miss | As held in Antulay, the competent authority before granting sanction has to apply its mind and be satisfied ab... |
| 23 | 759 | `0.5272` | `0.4248` | miss | I also fully appreciate that if the executive follows this rule strictly, a further protection from harassment... |
| 24 | 1092 | `0.5268` | `0.4425` | miss | It was also suggested at the hearing that the absence of need of sanction for prosecution under Section 6 of t... |
| 25 | 1101 | `0.5261` | `0.9123` | miss | The concept of sanction for prosecution by a superior is so inextricably woven into the fabric of the enactmen... |
| 26 | 1028 | `0.5257` | `0.3922` | miss | Not to have a corrupt Judge or a Judge who has misbehaved is unquestionably in public inter est but at the sam... |
| 27 | 1045 | `0.5247` | `0.5789` | miss | Clause (5) of Article 124 enables enactment of a special law by the Parliament to regulate the procedure for p... |
| 28 | 328 | `0.5232` | `0.5405` | miss | He squarely falls within the purview of the Act provided the second requirement under clause (c) of Section 6(... |
| 29 | 1094 | `0.5227` | `0.9195` | miss | The need for sanction under Section 6 for prosecution of the holder of a public office indicates the ambit and... |
| 30 | 1057 | `0.5217` | `0.8485` | miss | Construction of Section 6(1)(c) of the Act as suggested by the learned Solicitor General by treating the Presi... |
| 31 | 1056 | `0.5215` | `1.0` | MATCH | The need for these special provisions is a clear pointer in the direction of inapplicability to them of the ge... |
| 32 | 1050 | `0.5214` | `0.8312` | miss | There is no escape from the conclusion that Article 124(5) is wide enough to include within its ambit every co... |
| 33 | 1113 | `0.5211` | `0.4667` | miss | A charge of corruption against a superior Judge amounting to criminal misconduct by abuse of his office would ... |
| 34 | 827 | `0.5208` | `0.5068` | miss | He argued that the special provisions in the Constitution of India relating to the Judges of the High Courts a... |
| 35 | 1169 | `0.5206` | `0.3689` | miss | The duty and credit for maintaining this high tradition is on the Government in existence when the 'Constituti... |
| 36 | 1137 | `0.5206` | `0.3415` | miss | It is, therefore, time that all the constitutional functionaries involved in the process of appointment of sup... |
| 37 | 1016 | `0.5202` | `0.8421` | miss | Obviously, the competent sanctioning authority envisaged thereby is a vertical superior in the hierarchy havin... |
| 38 | 115 | `0.5199` | `0.4127` | miss | Section 6 creates a bar to the court from taking cogni zance of offences therein enumerated except with the pr... |
| 39 | 52 | `0.5197` | `0.5376` | miss | The third most crucial question that fails for consider ation in this case is who is the competent authority t... |
| 40 | 1125 | `0.5196` | `0.4167` | miss | Maybe, need is now felt for a law providing for trial and punishment of a superior Judge who is charged with t... |
| 41 | 1102 | `0.5188` | `1.0` | MATCH | The clear legislative intent is that the enactment applies only to those in whose case sanction of this kind i... |
| 42 | 982 | `0.5183` | `0.5051` | miss | Therefore, it unquestionably follows that the sanction to prosecute can be given by an authority competent to ... |
| 43 | 956 | `0.5163` | `0.7838` | miss | Admittedly, such previous sanction is a condition precedent for taking cognizance of an offence punishable und... |
| 44 | 1063 | `0.5161` | `0.475` | miss | The decision in S.P. Gupta was rendered much later and while dealing with the situations arising out of allega... |
| 45 | 75 | `0.5159` | `0.5` | miss | The has been enacted by Parliament to regulate 224 the procedure for the investigation and proof of the misbe ... |
| 46 | 1183 | `0.5159` | `0.931` | miss | Clearly, it was expected that the higher judiciary whose word would be final in the interpretation of all laws... |
| 47 | 1115 | `0.5155` | `0.4872` | miss | There is no escape from the conclusion that the gross misbehaviour of corruption of a Judge must undoubtedly f... |
| 48 | 1072 | `0.5146` | `0.9114` | miss | In order to form an objective opin ion, the competent authority must undoubtedly have before it the version of... |
| 49 | 1034 | `0.5142` | `0.4255` | miss | As the law now stands it is not open to any single individual, whether it is the President or the Chief Justic... |
| 50 | 1136 | `0.514` | `0.3939` | miss | The need for such legislation now would, therefore, not be entirely on account of the absence of it so far, bu... |
| 51 | 1010 | `0.5137` | `0.44` | miss | Since the order of removal in such a case is to be made by the President, the learned Solicitor General conten... |
| 52 | 16 | `0.5113` | `0.8211` | miss | So to initiate a proceeding against a Judge of a Supreme Court for criminal misconduct failing under Section 5... |
| 53 | 963 | `0.5112` | `0.4427` | miss | This appears to be the obvious conclusion even for a case like the present where no such sanction for prosecut... |
| 54 | 1013 | `0.5111` | `0.8923` | miss | Section 6(1)(c) speaks of 'authority competent to remove ' which plainly indicates the substantive competence ... |
| 55 | 991 | `0.511` | `0.4615` | miss | That is why the Legislature clearly provided that that authority alone would be competent to grant sanction wh... |
| 56 | 1107 | `0.5109` | `0.8217` | miss | If there is now a felt need to provide for such a situation, the remedy lies in suitable parliamentary legisla... |
| 57 | 1047 | `0.5108` | `0.5312` | miss | It is significant that clause (5) of Article 124 covers the field of 'investiga tion ' and 'proof ' of the 'mi... |
| 58 | 1134 | `0.5106` | `0.4` | miss | If this scheme is found to be inadequate in the present context, it is also indicative of the failure of the c... |
| 59 | 699 | `0.5104` | `0.4054` | miss | It has been urged that in view of this essential requirement it has to be held that the Act does not cover the... |

