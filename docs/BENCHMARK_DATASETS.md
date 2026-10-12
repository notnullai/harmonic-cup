# Benchmark datasets: licences and use

What each public dataset may be used for, checked before anything is downloaded
(HARM-S9-T12). The strategy's rule: evaluation and publication may use any
dataset whose terms allow it; the machine families that ship are trained only
on data that allows commercial use.

| Dataset | Licence as found | Evaluate | Publish results | Train what ships | Checked |
|---|---|---|---|---|---|
| **NASA IMS** (4 bearing failures run to failure) | data.gov lists it as a U.S. Government Work (public domain), public access. The data were donated by the University of Cincinnati's IMS centre, and the repository asks publications to acknowledge the repository and the donors | Yes | Yes, with the acknowledgement | **Yes, per its catalogue listing.** The donor was a university, not NASA, so keep the acknowledgement and re-check if a customer contract asks for provenance | 2026-09-29 |
| **CWRU Bearing Data Center** (seeded faults) | No licence or terms found on the data center's site. Kaggle mirrors carry their own licences, which don't bind the original | Yes | Yes, with the customary citation | **Not until CWRU confirms.** Ask the Case School of Engineering (the site lists a contact) before training anything that ships | 2026-09-29 |
| **XJTU-SY** (15 bearings run to failure) | No licence file. The README says the data are publicly available, that "anyone can use them to validate prognostics algorithms", and requests a citation of Wang et al. (2020) | Yes | Yes, with the citation | **No**: commercial use is not granted. Ask the author (listed in the README) if it is ever needed | 2026-09-29 |
| **FEMTO/PRONOSTIA** (PHM 2012 Challenge, 17+11 bearings run to failure) | No licence on the FEMTO-ST publiweb page. A widely-used GitHub mirror of the dataset says outright: "I don't know if there was any license to this," and asks only for the Nectoux et al. (2012) citation | Yes | Yes, with the citation | **Ask.** No licence stated anywhere found; ask FEMTO-ST / the PHM Society before training anything that ships | 2026-09-29 |
| **Paderborn KAt-DataCenter** (seeded + real bearing damage, vibration + motor current) | CC BY-NC 4.0, stated on the KAt-DataCenter page (mb.uni-paderborn.de/kat/forschung/bearing-datacenter): noncommercial academic use only, with a citation of Lessmeier et al.; commercial use requires contacting the author | Yes | Yes, with the citation | **No**: an NC licence | 2026-09-29 |
| **MaFaulDa** (UFRJ rotating-machine simulator, 6 fault classes) | No licence file on the UFRJ SMT page (www02.smt.ufrj.br/~offshore/mfs/); it asks only for a citation of Ribeiro (2016) and gives an author e-mail for licensing questions | Yes | Yes, with the citation | **Ask.** No licence found; ask UFRJ / Felipe Ribeiro before training anything that ships | 2026-09-29 |
| **MFPT** (Society for Machinery Failure Prevention Technology bearing/gear set) | No terms confirmed: mfpt.org/fault-data-sets/ now 301-redirects to asnt.org/about/managed-affiliates/mfpt/, which no longer links to the dataset or states any terms. Copy used (2026-10-03): the MathWorks mirror, github.com/mathworks/RollingElementBearingFaultDiagnosis-Data (the 20 rig recordings), which states CC BY-NC-SA 4.0, sources data-acoustics.com and names the owner, Eric Bechhoefer, for any commercial use | Yes, if a copy can still be obtained, with the customary citation | Yes, with the citation | **Ask.** The official page has moved and states no terms; confirm with MFPT/ASNT before training anything that ships | 2026-09-29 |
| **MetroPT** and **MetroPT-3** (Porto metro compressor air-production unit) | CC BY 4.0, stated on both the UCI ML Repository MetroPT-3 page (archive.ics.uci.edu/dataset/791) and the Zenodo record for the original MetroPT (zenodo.org/records/6854240) | Yes | Yes | **Yes**: CC BY | 2026-09-29 |
| **MIMII** (Hitachi acoustic dataset, valves/pumps/fans/slide rails) | CC BY-SA 4.0, stated on the original MIMII Zenodo record (zenodo.org/records/3384388). The later MIMII DUE and MIMII DG follow-ons are CC BY-NC-SA 4.0 instead | Yes | Yes, with the citation | **Ask.** ShareAlike is not on the clear-yes list (CC BY / CC0 / public domain / U.S. Government Work): a model trained on it may have to carry the same licence. Confirm before shipping | 2026-09-29 |
| **DCASE Task 2** (Hitachi/NTT machine-sound anomaly detection, multi-year) | CC BY-NC-SA 4.0, stated in the Zenodo record's "Condition of use" (checked on the 2023 development set, zenodo.org/records/7882613); consistent across the challenge's other years | Yes | Yes, with the citation | **No**: an NC licence | 2026-09-29 |

## XJTU-SY

- **Source:** [github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets). The downloads it lists are the author's site, Google Drive, Dropbox, MediaFire, MEGA and Baidu. The size is not stated there; check it before asking the owner to approve the download.
- **Cite:** B. Wang, Y. Lei, N. Li, N. Li, "A Hybrid Prognostics Approach for Estimating Remaining Useful Life of Rolling Element Bearings", *IEEE Transactions on Reliability*, 69(1), 401–412, 2020.
- **Why it fits T15:** the failed element is labelled per bearing (outer race, inner race, cage, and mixed), so it can confirm the bearing-location refiner on data never used to set it.
- **Downloaded** 2026-09-29, with the owner's approval, into `datasets/xjtu-sy/` (gitignored):
  - the six-part archive, 4.44 GB, each part checked against MediaFire's SHA-256;
  - the authors' two reference papers, in `references/`.
- **Unpacking:** Windows `tar` reads each part on its own (WSL's 7-Zip has no RAR decoder). All 9217 files are present, 12.2 GB. **Five files that span two parts are damaged**, part null-padded, and must be skipped:
  - `37.5Hz11kN/Bearing2_3/358.csv`
  - `40Hz10kN/Bearing3_1/966.csv`, `1832.csv`
  - `40Hz10kN/Bearing3_2/2317.csv`
  - `40Hz10kN/Bearing3_4/1105.csv`
- **Facts from the authors' own write-up** (Lei et al., *Journal of Mechanical Engineering* 55(16), 2019, Tables 1–3):
  - bearing LDK UER204: 8 balls, ball diameter 7.92 mm, pitch diameter 34.55 mm, contact angle 0°;
  - conditions: 1 = 2100 rpm at 12 kN, 2 = 2250 rpm at 11 kN, 3 = 2400 rpm at 10 kN;
  - sampling 25.6 kHz, a 1.28 s snapshot every minute; each `N.csv` holds column 1 horizontal (the load direction), column 2 vertical;
  - a test ends when the amplitude passes 10× its normal maximum.

| Bearing | Snapshots | Failed element | Bearing | Snapshots | Failed element |
|---|---|---|---|---|---|
| 1_1 | 123 | outer race | 2_4 | 42 | outer race |
| 1_2 | 161 | outer race | 2_5 | 339 | outer race |
| 1_3 | 158 | outer race | 3_1 | 2538 | outer race |
| 1_4 | 122 | cage | 3_2 | 2496 | inner race, rolling element, cage, outer race |
| 1_5 | 52 | inner and outer race | 3_3 | 371 | inner race |
| 2_1 | 491 | inner race | 3_4 | 1515 | inner race |
| 2_2 | 161 | outer race | 3_5 | 114 | outer race |
| 2_3 | 533 | cage | | | |

- **Use:** evaluation and the public benchmark only.

## What the other datasets add, and their size

| Dataset | What it adds | Download size |
|---|---|---|
| FEMTO/PRONOSTIA | A third independent run-to-failure vibration benchmark (different rig, conditions) alongside IMS and XJTU-SY, for cross-dataset generalization checks | Not stated on the official page |
| Paderborn KAt | Real (not only seeded) bearing damage, plus a motor-current channel alongside vibration | Not stated on the KAt-DataCenter page |
| MaFaulDa | A rotating-machine simulator with imbalance and misalignment faults, broadening the fault vocabulary beyond bearings | 13.0 GB (.tgz or .zip) |
| MFPT | An independently curated, widely-cited bearing set with real wind-turbine field failures alongside lab-seeded faults | Not stated (official page has moved) |
| MetroPT / MetroPT-3 | A non-bearing, real industrial asset (a metro compressor's air-production unit) with real, timestamped failures — a different machine class entirely | MetroPT-3 (UCI): 208.2 MB. MetroPT (Zenodo): 1.6 GB |
| MIMII | Acoustic (microphone) signals for four industrial machine types — a different sensor modality than vibration | ~100 GB (all machine types and SNR mixes) |
| DCASE Task 2 | An actively maintained, multi-year acoustic anomaly-detection benchmark with domain-shift structure, complementing MIMII | 4.8 GB (2023 development set) |
