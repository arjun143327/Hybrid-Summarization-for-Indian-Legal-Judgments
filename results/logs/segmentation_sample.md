# Sentence Segmentation Validation Report (25 Sample Judgments)

This report compares naive NLTK sent_tokenize against our updated citation-, decimal-date-, and case-insensitive abbreviation-protected segmentation (segment_sentences) across 25 randomly sampled Indian Supreme Court judgments.

**Random Seed for Sampling**: 42  
**Sample Size**: 25 judgments  

**Total Naive Sentences Across 25 Docs**: 4,058  
**Total Protected Sentences Across 25 Docs**: 3,623  
**Net False Splits Prevented**: 435 (10.72% reduction)  

---

---

## Sample 1: Case ID 5243
- **Raw text character length**: 14,380
- **Naive sent_tokenize count**: 98
- **Citation-protected count**: 90
- **Count difference**: 8 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] Civil Appeal No. 123(N) of 1973.
[2] From the Judgment and Order dated 26.
[3] 1972 of the Andhra Pradesh High Court in Writ Appeal No. 444 of 1968.
[4] B. Parthasarthy and G .N.
Rao for the Appellant.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) Civil Appeal No. 123(N) of 1973.
(Example 2) 1972 of the Andhra Pradesh High Court in Writ Appeal No. 444 of 1968.
(Example 3) The facts giving rise to the writ petition instituted by the 72 allottees to whom the houses were allotted need to be stated briefly: The Hyderabad Municipal Corporation started a scheme called Low Income Housing Scheme in 1957.
`

---

## Sample 2: Case ID 914
- **Raw text character length**: 27,294
- **Naive sent_tokenize count**: 170
- **Citation-protected count**: 153
- **Count difference**: 17 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] Civil Appeal No. 2329 of 1977.
[2] Appeal by Special Leave from the Judgment and Order dated 30 11 1976 of the Karnataka High Court in W.P.
No 2307/71.
[3] CIVIL APPEAL NOS.
2330 2350/77 Appeals by Special Leave from the Judgment and Order dated 30 11 1976 of the Karnataka High Court in W.P. Nos.
2307/71, 796/72, and 462 467, 553 560, 943, 944, 1033, 1027 and 1032/73; and CIVIL APPEAL NOS.
2351 2370/77 ...
[4] P. Ram Reddy and section section Javali for the Appellant in CA 2329/77.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) Civil Appeal No. 2329 of 1977.
(Example 2) Appeal by Special Leave from the Judgment and Order dated 30 11 1976 of the Karnataka High Court in W.P.
No 2307/71.
(Example 3) CIVIL APPEAL NOS.
2330 2350/77 Appeals by Special Leave from the Judgment and Order dated 30 11 1976 of the Karnataka High Court in W.P. Nos.
2307/71, 796/72, and 462 467, 553 560, 943, 944, 1033, 1027 and 1032/73; and CIVIL APPEAL NOS.
2351 2370/77 Appeals by Special Leave from the Judgment and Ord...
`

---

## Sample 3: Case ID 205
- **Raw text character length**: 18,358
- **Naive sent_tokenize count**: 101
- **Citation-protected count**: 91
- **Count difference**: 10 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] vil Appeal No. 160 of 1950.
[2] Appeal against the judgment and Decree dated the 30th March, 1951, of the High Court of judicature at Bombay (Chagla C. J. and Tendolkar J.) in Income Tax Reference No. 34 of 1950.
[3] C. K. Daphtary, Solicitor General for India, (Porus A. Mehta, with him) for the appellant.
[4] R. J. Kolah for the respondent.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) vil Appeal No. 160 of 1950.
(Example 2) Appeal against the judgment and Decree dated the 30th March, 1951, of the High Court of judicature at Bombay (Chagla C. J. and Tendolkar J.) in Income Tax Reference No. 34 of 1950.
(Example 3) (2) A.I.R. 1950 Cal.
452 Profits Tax, Madras,v,.
`

---

## Sample 4: Case ID 6079
- **Raw text character length**: 36,513
- **Naive sent_tokenize count**: 165
- **Citation-protected count**: 146
- **Count difference**: 19 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] vil Appeal No. 612 (NT) of 1975.
[2] From the Judgment and Order dated 24/25.9.1974 of the Gujarat High Court in Special Civil Application No. 1797 of 1972.
[3] Harish N. Salve, Mrs. A.K. Verma and Joel Pares for the Appellant.
[4] V.S. Desai, M.B. Rao and Ms. A. Subhashini for the Respondents.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) vil Appeal No. 612 (NT) of 1975.
(Example 2) From the Judgment and Order dated 24/25.9.1974 of the Gujarat High Court in Special Civil Application No. 1797 of 1972.
(Example 3) The Judgment of the Court was delivered by KANIA, J.
This is an appeal from the judgment of a Division of 5 the High Court of Gujarat in Special Civil Application No. 1797 of 1972 on a certificate granted under Article 133(1) of the Constitution of India.
`

---

## Sample 5: Case ID 2255
- **Raw text character length**: 16,008
- **Naive sent_tokenize count**: 112
- **Citation-protected count**: 106
- **Count difference**: 6 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] Appeal NO.
437 of 1966.
[2] Appeal by special leave from the Award dated March 31, 1964 of the Industrial Tribunal, Maharashtra in Reference (IT) No. 40 of 1963.
[3] H. K. Sowani, K. Rajendra Chaudhuri and K. R. Chaudhuri, for the appellant.
[4] H. R. Gokhale and 1.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) Appeal NO.
437 of 1966.
(Example 2) Appeal by special leave from the Award dated March 31, 1964 of the Industrial Tribunal, Maharashtra in Reference (IT) No. 40 of 1963.
(Example 3) N. Shroff, for respondent No. 1.
`

---

## Sample 6: Case ID 2008
- **Raw text character length**: 15,788
- **Naive sent_tokenize count**: 97
- **Citation-protected count**: 67
- **Count difference**: 30 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] Appeal No. 879 of 1962 etc.
Appeals by special leave from the judgment and decrees dated January 18, 1961, and December 13, 1960 of the Punjab High Court Circuit Bench at Delhi, in Civil Revision No. 13 D of 1958 and Civil Revision Case No. 592 D of ...
[2] M.S.K. Sastri and M. section Narasimhan, for the appellant (in C.A. No. 121/63) M. C. Setalvad, section Murty and B. P. Maheshwari, for the appellants (in C.A. No. 879 of 1962) and respondents (in C.A. No. 121 of 1962) Raghbir Singh and M. I. Khowaja...
[3] The Judgment of the Court was delivered by Wanchoo, J.
These two appeals by special leave from two judgments of the Punjab High Court raise a common question with respect to the application of the first proviso to section 57 (2) of the Delhi Rent Con...
[4] They arise from decisions of two learned Single Judges in revision applications under the Delhi and Ajmer Rent Control Act, No. 38 of 1952 (hereinafter referred to as the 1952 Act.)
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) Appeal No. 879 of 1962 etc.
Appeals by special leave from the judgment and decrees dated January 18, 1961, and December 13, 1960 of the Punjab High Court Circuit Bench at Delhi, in Civil Revision No. 13 D of 1958 and Civil Revision Case No. 592 D of 1957.
(Example 2) M.S.K. Sastri and M. section Narasimhan, for the appellant (in C.A. No. 121/63) M. C. Setalvad, section Murty and B. P. Maheshwari, for the appellants (in C.A. No. 879 of 1962) and respondents (in C.A. No. 121 of 1962) Raghbir Singh and M. I. Khowaja, for respondent (in C.A. No. 879 of 1962).
(Example 3) The Judgment of the Court was delivered by Wanchoo, J.
These two appeals by special leave from two judgments of the Punjab High Court raise a common question with respect to the application of the first proviso to section 57 (2) of the Delhi Rent Control Act, No. 59 of 1958, (hereinafter referred to...
`

---

## Sample 7: Case ID 1830
- **Raw text character length**: 10,914
- **Naive sent_tokenize count**: 96
- **Citation-protected count**: 90
- **Count difference**: 6 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] minal Appeal No. 193 of 1961.
[2] Appeal by special leave from the judgment and order dated February 9 and 10, 1961, of the Gujarat High Court in Criminal Appeal No. 367 of 1960.
[3] D.R. Prem, K.L. Hathi and R.H. Dhebar.
[4] for the appellant.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) minal Appeal No. 193 of 1961.
(Example 2) Appeal by special leave from the judgment and order dated February 9 and 10, 1961, of the Gujarat High Court in Criminal Appeal No. 367 of 1960.
(Example 3) December 6, 1963.
`

---

## Sample 8: Case ID 1145
- **Raw text character length**: 44,867
- **Naive sent_tokenize count**: 324
- **Citation-protected count**: 308
- **Count difference**: 16 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] Appeal No. 759 of 1957.
[2] Appeal from the judgment and Order dated June 26, 1957, of the Bombay High Court in Appeal No, 92 of 1956.
[3] J. C. Bhatt, section N. Andley, J. B. Dadachanji, Rameshuar Nath and.
[4] P. L, Vohra, for the appellants.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) Appeal No. 759 of 1957.
(Example 2) Appeal from the judgment and Order dated June 26, 1957, of the Bombay High Court in Appeal No, 92 of 1956.
(Example 3) WANCHOO, J.
This appeal by certificate granted by the High Court of Bombay raises the constitutionality, of section 114(2) of the Bombay Industrial Relations Act, No. XI of 1947, hereinafter called the Act).
`

---

## Sample 9: Case ID 6038
- **Raw text character length**: 21,548
- **Naive sent_tokenize count**: 206
- **Citation-protected count**: 190
- **Count difference**: 16 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] vil Appeal Nos.
2704 06 of 1979.
[2] From the Judgment and Order dated 1.5.1979 of the Guja rat High Court in Special Civil Appln.
[3] Nos. 133 of 1976, 325 and 384 of 1976.
[4] A.B. Rohatagi, Harish N. Salve, Ms. Palavi Shroff, S.S. Shroff, P.S. Shroff and R. Sasiprabhu for the Appellants.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) vil Appeal Nos.
2704 06 of 1979.
(Example 2) From the Judgment and Order dated 1.5.1979 of the Guja rat High Court in Special Civil Appln.
(Example 3) Nos. 133 of 1976, 325 and 384 of 1976.
`

---

## Sample 10: Case ID 841
- **Raw text character length**: 14,211
- **Naive sent_tokenize count**: 121
- **Citation-protected count**: 113
- **Count difference**: 8 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] Civil Appeal No. 355 of 1958.
[2] Appeal by special leave from the decision dated December 12, 1956, of the Labour Appellate Tribunal 983 of India, Bombay in Appeal (Bom.) Nos.
77and 103 of 1956.
[3] A. V. Viswanatha Sastri, section N. Andley, J. B. Dadachanji and Rameshwar Nath for the appellant.
[4] B. D. Sharma, for respondent No. 1.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) Civil Appeal No. 355 of 1958.
(Example 2) Appeal by special leave from the decision dated December 12, 1956, of the Labour Appellate Tribunal 983 of India, Bombay in Appeal (Bom.) Nos.
77and 103 of 1956.
(Example 3) B. D. Sharma, for respondent No. 1.
`

---

## Sample 11: Case ID 5548
- **Raw text character length**: 4,888
- **Naive sent_tokenize count**: 61
- **Citation-protected count**: 36
- **Count difference**: 25 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] tition (Criminal) Nos. 225 and 513 of 1987.
[2] (Under Article 32 of the Constitution of India).
[3] 41 L.K. Pandey for the petitioner in W.P. No.225 of 1987.
[4] M.S. Gupta for the petitioner in W.P. No. 513 of 1987.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) tition (Criminal) Nos. 225 and 513 of 1987.
(Example 2) 41 L.K. Pandey for the petitioner in W.P. No.225 of 1987.
(Example 3) M.S. Gupta for the petitioner in W.P. No. 513 of 1987.
`

---

## Sample 12: Case ID 6072
- **Raw text character length**: 8,544
- **Naive sent_tokenize count**: 59
- **Citation-protected count**: 53
- **Count difference**: 6 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] vil Appeals Nos.
2 4 of 1975.
[2] From the Judgment and Order dated 8th 9th November 1973 of the Gujarat High Court in Estate Duty Reference Nos. 2, 3 and 4 of 1971.
[3] Dr. V. Gauri Shankar and Miss A. Subhashini for the Appel lant.
[4] V.S. Desai, Mrs. A.K. Verma and Joel Peres for the Respondents.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) vil Appeals Nos.
2 4 of 1975.
(Example 2) From the Judgment and Order dated 8th 9th November 1973 of the Gujarat High Court in Estate Duty Reference Nos. 2, 3 and 4 of 1971.
(Example 3) One Abdulhussein Gulamhussein Merchant died on 8 February, 1959.
`

---

## Sample 13: Case ID 4471
- **Raw text character length**: 20,786
- **Naive sent_tokenize count**: 132
- **Citation-protected count**: 119
- **Count difference**: 13 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] Civil Appeal No. 2132 of 1977.
[2] Appeal by special leave from the judgment and order dated the 23rd September, 1976 of the Gujarat High Court in First Appeal No. 76 of 1963 D.V. Patel, R. Shroff, Gopal Subramaniam and D.P. Mohanty for the Appellant.
[3] M.N. Phadke, S.C. Patel and R.N. Poddar for the Respondent.
[4] The Judgment of the Court was delivered by MISRA J.
The present appeal by special leave is directed against the Full Bench decision of the High Court of Gujarat at Ahmedabad dated 23rd of September, 1976.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) Civil Appeal No. 2132 of 1977.
(Example 2) Appeal by special leave from the judgment and order dated the 23rd September, 1976 of the Gujarat High Court in First Appeal No. 76 of 1963 D.V. Patel, R. Shroff, Gopal Subramaniam and D.P. Mohanty for the Appellant.
(Example 3) The Judgment of the Court was delivered by MISRA J.
The present appeal by special leave is directed against the Full Bench decision of the High Court of Gujarat at Ahmedabad dated 23rd of September, 1976.
`

---

## Sample 14: Case ID 714
- **Raw text character length**: 36,535
- **Naive sent_tokenize count**: 191
- **Citation-protected count**: 177
- **Count difference**: 14 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] Appeal No. 343 of 1955.
[2] Appeal from the judgment and order dated September 13, 1954, of the Patna High Court in Misc.
[3] Case No. 39 of 1954.
[4] L. K. Jha, B. K. P. Sinha and R. C. Prasad, for the appellant.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) Appeal No. 343 of 1955.
(Example 2) Appeal from the judgment and order dated September 13, 1954, of the Patna High Court in Misc.
(Example 3) Case No. 39 of 1954.
`

---

## Sample 15: Case ID 4842
- **Raw text character length**: 7,299
- **Naive sent_tokenize count**: 51
- **Citation-protected count**: 45
- **Count difference**: 6 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] l Leave Petition (Civil) No. 3786 of 1982.
[2] From the Judgement and Order dated the 21st April, 1978 of the Calcutta High Court in Income Tax Reference No. 573 of 1971.
[3] K.C. Dua and Miss A. Subhashini for the Petitioner.
[4] The Judgment of the Court was delivered by VENKATARAMIAH, J.
This Special Leave Petition is filed under Article 136 of the Constitution by the Commissioner of Income tax, West Bengal, Calcutta against the decision of the High Court of Calcutta in Inc...
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) l Leave Petition (Civil) No. 3786 of 1982.
(Example 2) From the Judgement and Order dated the 21st April, 1978 of the Calcutta High Court in Income Tax Reference No. 573 of 1971.
(Example 3) The Judgment of the Court was delivered by VENKATARAMIAH, J.
This Special Leave Petition is filed under Article 136 of the Constitution by the Commissioner of Income tax, West Bengal, Calcutta against the decision of the High Court of Calcutta in Income tax Reference No. 573 of 1971.
`

---

## Sample 16: Case ID 3459
- **Raw text character length**: 62,226
- **Naive sent_tokenize count**: 382
- **Citation-protected count**: 358
- **Count difference**: 24 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] Civil Appeals Nos. 1270, 1315 1316 of 1975.
[2] Appeals by Special Leave from the Judgment and Order dated the 26 8 75 of the Joint Judge at Thana in Election Petitions Nos. 3 and 4 of 1974.
[3] R. P. Bhat (In CAs.
1315 1316/75, K. R. Chaudhury, K. Rajendra Chaudhury and Mrs. Veena Khanna for the Appellants in CAs.
1315 1316/75 and in C.A.
[4] 1270/75.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) Civil Appeals Nos. 1270, 1315 1316 of 1975.
(Example 2) Appeals by Special Leave from the Judgment and Order dated the 26 8 75 of the Joint Judge at Thana in Election Petitions Nos. 3 and 4 of 1974.
(Example 3) D.V. Patel (In CAs.
1315 16/75, V. N. Ganpule for respondent No. 1 in all the appeals.
`

---

## Sample 17: Case ID 261
- **Raw text character length**: 29,474
- **Naive sent_tokenize count**: 196
- **Citation-protected count**: 177
- **Count difference**: 19 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] Appeal No. 108 of 1952.
[2] Appeal from the Judgment and Decree dated the 29th March, 1950, of the High Court of Judicature at 468 Calcutta in Appeal from Original Decree No. 121 of 1945 arising from the Decree dated the 22nd December, 1944, of the Court of Subordinate Judge at...
[3] N. C. Chatterjee (C. N. Laik, D. N. Mukherjee and Sukumar Ghose, with him) for the appellants.
[4] section P. Sinha (B.B. Haldar and section C. Bannerji, with him) for respondents Nos.
I to 3.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) Appeal No. 108 of 1952.
(Example 2) Appeal from the Judgment and Decree dated the 29th March, 1950, of the High Court of Judicature at 468 Calcutta in Appeal from Original Decree No. 121 of 1945 arising from the Decree dated the 22nd December, 1944, of the Court of Subordinate Judge at Alipore, in Title Suit No. 70 of 1941.
(Example 3) 1954.
`

---

## Sample 18: Case ID 245
- **Raw text character length**: 43,429
- **Naive sent_tokenize count**: 285
- **Citation-protected count**: 278
- **Count difference**: 7 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] 7 of 1953.
[2] Under article 32 of the Constitution of India, praying that the Order of the Governor of Uttar Pradesh dated the 29th August, 1952, revoking the grants made by the Rulers of Charkhari and Sarila in favour of the petitioners be declared void.
[3] K. section Krishna Swamy Iyengar, and section P. Sinha (Bishan Singh and section section Shukla, with them) for the petitioners.
[4] Gopalji Mehrotra and C. P. Lal for the respondent.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) 7 of 1953.
(Example 2) Under article 32 of the Constitution of India, praying that the Order of the Governor of Uttar Pradesh dated the 29th August, 1952, revoking the grants made by the Rulers of Charkhari and Sarila in favour of the petitioners be declared void.
(Example 3) The integration did not work satisfactorily, so, on 26th December, 1949, the same thirty five Rulers entered into another agreement abrogating their covenant and dissolving the newly created State as from 1ST January, 1950.
`

---

## Sample 19: Case ID 769
- **Raw text character length**: 5,432
- **Naive sent_tokenize count**: 41
- **Citation-protected count**: 35
- **Count difference**: 6 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] vil Appeal No. 310 of 1954.
[2] Appeal by special leave from the judgment and order dated March 22, 1956, of the Labour Appellate Tribunal of India, Calcutta in Appeal No. Cal.
183 of 1955.
[3] N.C. Chatterjee, section N. Mukherjee and B. N. Ghosh for the appellant.
[4] Sukumr Ghosh, for the respondent.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) vil Appeal No. 310 of 1954.
(Example 2) Appeal by special leave from the judgment and order dated March 22, 1956, of the Labour Appellate Tribunal of India, Calcutta in Appeal No. Cal.
183 of 1955.
(Example 3) During the pendency of that dispute, the appellant laid off the respondent who was an employee in the ration shop maintained by the appellant from July 19, 1954, as rationing of food stuff came to an end from July 10, 1954.
`

---

## Sample 20: Case ID 1793
- **Raw text character length**: 11,350
- **Naive sent_tokenize count**: 88
- **Citation-protected count**: 77
- **Count difference**: 11 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] Appeal No.241 of 1961.
[2] Appeal from the judgment and decree dated March 4, 1958, of the Patna High Court in Appeal from Appellate Decree No. 1335 of 1952.
[3] 634 R.S. Sinha and R.C. Prasad, for the appellants.
[4] Sarjoo Prasad and B. P. Jha, for the respondents nos.
1 and 2.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) Appeal No.241 of 1961.
(Example 2) Appeal from the judgment and decree dated March 4, 1958, of the Patna High Court in Appeal from Appellate Decree No. 1335 of 1952.
(Example 3) April 3, 1964.
`

---

## Sample 21: Case ID 1907
- **Raw text character length**: 32,432
- **Naive sent_tokenize count**: 215
- **Citation-protected count**: 171
- **Count difference**: 44 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] ivil Appeal No. 875 of 1964.
[2] Appeal by special leave from the judgment and order dated April 10, 1964, of the Calcutta High Court in Civil Rule No. 4439 of 1962.
[3] N.C. Chatterjee and D. Goburdhan, for the appellants.
[4] P.K. Chatterjee and D.N. Mukherjee, for the respondent.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) ivil Appeal No. 875 of 1964.
(Example 2) Appeal by special leave from the judgment and order dated April 10, 1964, of the Calcutta High Court in Civil Rule No. 4439 of 1962.
(Example 3) The Judgment of the Court was delivered by Gajendragadkar, C.J.
Appellant No. 1, Kaluram Onkarmal, was let into possession of the premises described as holding No. 182H, G.T. Road, Asansol as a monthly tenant under Harbhajan Singh Wasal who was the owner of the said premises.
`

---

## Sample 22: Case ID 4143
- **Raw text character length**: 4,007
- **Naive sent_tokenize count**: 35
- **Citation-protected count**: 30
- **Count difference**: 5 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] Civil Appeal No. 174 of 1976.
[2] Appeal by Special Leave from the Judgment and order dated 1 11 1974 of the Delhi High Court in L.P.A. No. 19/71.
[3] P. P. Rao, A. K. Ganguli and R. Venkataramani for the Appellant.
[4] T. A. Francis and Miss A. Subhashini for the Respondents.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) Civil Appeal No. 174 of 1976.
(Example 2) Appeal by Special Leave from the Judgment and order dated 1 11 1974 of the Delhi High Court in L.P.A. No. 19/71.
(Example 3) Rules 1965.
`

---

## Sample 23: Case ID 4936
- **Raw text character length**: 65,522
- **Naive sent_tokenize count**: 415
- **Citation-protected count**: 353
- **Count difference**: 62 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] l Appeal No. 4803 of 1984 From the Judgment and Order dated 7.8.84 of the Allahabad High Court in Civil Misc.
[2] Application No. 10968 of 84 & S.A. No. 2182.
[3] K.K. Venugopal, R N. Karanjawala & Mrs. Manik Karanjawala for the appellant.
[4] R.Parasaran, Attorney General of India.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) l Appeal No. 4803 of 1984 From the Judgment and Order dated 7.8.84 of the Allahabad High Court in Civil Misc.
(Example 2) Application No. 10968 of 84 & S.A. No. 2182.
(Example 3) Ashok Desai, Anil Diwan Pinaki Mishra and Praveen Kumar for respondent No. 1,.
`

---

## Sample 24: Case ID 218
- **Raw text character length**: 32,970
- **Naive sent_tokenize count**: 195
- **Citation-protected count**: 189
- **Count difference**: 6 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] Civil Appeal No. 70 of 1952.
[2] Appeal by special leave from the Judgment and Decree dated the 5th May, 1949, of the High Court of Judicature at Patna (Manohar Lall and Mahabir Prasad JJ.) in Appeal from Appellate Decree No. 2091 of 1946.
[3] C.K. Daphtary, Solicitor General far India (G. N. Joshi and Porus A. Mehta, with him) for the appellant.
[4] P. Sinha (Nuruddin Ahmed, with him) for the respondent.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) Civil Appeal No. 70 of 1952.
(Example 2) Appeal by special leave from the Judgment and Decree dated the 5th May, 1949, of the High Court of Judicature at Patna (Manohar Lall and Mahabir Prasad JJ.) in Appeal from Appellate Decree No. 2091 of 1946.
(Example 3) No. 38, the Government of the Colony is liable to be sued in an action of tort as well as in contract.
`

---

## Sample 25: Case ID 4601
- **Raw text character length**: 41,016
- **Naive sent_tokenize count**: 222
- **Citation-protected count**: 171
- **Count difference**: 51 false splits prevented

### First 4 Segmented Sentences:
`	ext
[1] tition (Crl.)
[2] No. 850 of 1982 (Under article 32 of the Constitution of India.)
[3] S.B. Malik and K.B. Rohtagi for the Petitioner.
[4] Harbans Singh and D.D. Sharma for the Respondents.
`

### Spot-checked Protected Citation/Date Sentences:
`	ext
(Example 1) No. 850 of 1982 (Under article 32 of the Constitution of India.)
(Example 2) The petitioner was originally sentenced to death on 18.1.1969 by the learned Sessions Judge, Ferozepore, for committing an offence of murder under section 302 Indian Penal Code.
(Example 3) 1973 by the appropriate Government, with the result that he is liable to serve his sentence until the remainder of his life in prison under the ruling of this Court in Gopal Godse 's(1) case.
`

---
