# BloodWise AI — AI-Powered Blood Report Analyzer

Upload a blood report as a PDF or a photograph. BloodWise AI reads the test
values, compares each one with the right reference interval, explains in plain
language what an unusual result may be associated with, checks whether the
*combination* of values is uncommon, and produces a downloadable PDF summary.

**This is an educational screening and information tool. It is not a medical
diagnostic system. It does not diagnose, and it never names a medicine or a
dose. Every result should be discussed with a qualified healthcare
professional.**

B.Tech CSE one-credit project (PBL-III) · Abdul Sami Siddiqui (2427030061) and
Aditya Singh (2427030054) · Supervisor: Dr. Shashvat · Department of Computer
Science and Engineering, Manipal University Jaipur.

---

## What makes this more than a lookup table

Three design decisions shape the whole project.

**No medical number is written from memory.** Every reference interval,
explanation and unit conversion factor comes from a source that was fetched and
read while building the project, and every row records its source URL and the
date it was read. Where no source could be found — blood urea as distinct from
BUN, and random (non-fasting) glucose — the parameter is reported with its value
and *no judgement*, rather than being given a plausible-looking number. Two of
the 39 supported parameters are in that state, and the application says so on
screen.

**The laboratory's own interval wins.** Reference intervals differ between
laboratories, methods, ages and sexes, so a single universal table would be
wrong. Where the uploaded report prints its own interval, that one is used.
Otherwise the stored interval is used. Which of the two applied is shown next to
every result and in the PDF.

**Nothing is analysed until the user confirms it.** Values read from a file can
be wrong, so the application shows an editable table first, marks low-confidence
readings, and explains what is uncertain about each one.

---

## Features

- Reads text PDFs, scanned PDFs and photographs (JPG, PNG)
- Handles both PDF table layouts: whole rows on one line, and one cell per line
- Recognises 39 parameters across 8 panels through an alias list
- Converts SI units to conventional units using factors derived from data, and
  refuses to convert when it cannot justify a factor
- Flags implausible values, likely from a misread decimal point
- Mandatory review-and-correct step before any analysis
- Sex-specific reference intervals where the source provides them
- Association rules over patterns, each citing its sources
- Two anomaly detectors over the whole panel, with a measured comparison
- Retrieval over a MedlinePlus knowledge base, with citations on every claim
- Optional AI rewriting that is checked before it is shown, and is never required
- Interactive charts and a downloadable PDF report
- 80 automated tests that run without network access and without an API key

## Panels and parameters

| Panel | Parameters |
|---|---|
| Complete Blood Count | Hemoglobin, RBC, WBC, Platelets, Hematocrit, MCV, MCH, MCHC, RDW |
| Vitamin Profile | Vitamin B12, Vitamin D (25-OH), Folate |
| Thyroid Profile | TSH, T3 total, T4 total, Free T3, Free T4 |
| Diabetes Profile | Fasting glucose, Random glucose\*, HbA1c |
| Lipid Profile | Total cholesterol, HDL, LDL, Triglycerides |
| Kidney Function | Creatinine, BUN, Urea\*, eGFR, Uric acid |
| Liver Function | ALT, AST, ALP, Total bilirubin, Direct bilirubin, Albumin |
| Iron Profile | Ferritin, Serum iron, TIBC, Transferrin saturation |

\* No verified reference interval was found, so these are reported without a
judgement. Adding one means adding a row to `data/reference_ranges.csv` with its
source; no code changes.

---

## Installation

Python 3.10–3.12 is recommended.

```bash
git clone <your-repo-url>
cd BloodWise-AI
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### Tesseract OCR (needed only for images and scanned PDFs)

Tesseract is a separate program. `pip install pytesseract` installs only the
Python wrapper, which is the most common setup mistake.

| Platform | Command |
|---|---|
| Windows | Install from <https://github.com/UB-Mannheim/tesseract/wiki>, then set `TESSERACT_CMD` in `.env` to the full path of `tesseract.exe` |
| macOS | `brew install tesseract` |
| Ubuntu / Debian | `sudo apt install tesseract-ocr` |

### Poppler (needed only to OCR scanned PDFs)

`pdf2image` renders PDF pages using Poppler.

| Platform | Command |
|---|---|
| Windows | Download Poppler, add its `bin` folder to `PATH` |
| macOS | `brew install poppler` |
| Ubuntu / Debian | `sudo apt install poppler-utils` |

Text PDFs work without either of these.

### Optional settings

```bash
cp .env.example .env
```

`OPENAI_API_KEY` turns on AI rewriting. **It is entirely optional**: without it
the application uses its rule-based explanations, which are complete on their
own and are what the automated tests check.

---

## Running it

```bash
python scripts/build_reference_data.py     # writes the data files
python scripts/make_sample_reports.py      # writes the sample reports
streamlit run app.py
```

Then open <http://localhost:8501> and try a sample report from the dropdown on
the Upload page.

To rebuild the model from the NHANES release (see *Dataset* below):

```bash
python scripts/prepare_nhanes.py --nhanes-dir /path/to/nhanes_export
python scripts/train_model.py
```

The application runs without the model; the unusual-combination section simply
reports that it is unavailable.

---

## Project structure

```
BloodWise-AI/
├── app.py                        Streamlit application, 7 pages
├── modules/
│   ├── pdf_extractor.py          PDF text layer, falls back to OCR
│   ├── image_ocr.py              Tesseract with image preprocessing
│   ├── parameter_extractor.py    alias matching, values, units, intervals
│   ├── data_normalizer.py        unit handling and conversion
│   ├── reference_analyzer.py     intervals and the low/normal/high decision
│   ├── ml_analyzer.py            Mahalanobis + Isolation Forest
│   ├── rag_engine.py             retrieval over the knowledge base
│   ├── ai_explainer.py           association rules, explanations, LLM layer
│   └── report_generator.py       PDF output
├── dashboard/
│   ├── results.py                display shaping (pure, tested)
│   └── charts.py                 Plotly figures
├── data/
│   ├── reference_ranges.csv      44 rows, each with its source
│   ├── association_rules.csv     20 rules, each with its sources
│   ├── unit_conversions.csv      13 factors derived from NHANES
│   └── nhanes_training.csv       built by prepare_nhanes.py
├── knowledge_base/               39 MedlinePlus documents + SOURCES.md
├── scripts/
│   ├── build_reference_data.py   writes the three data files above
│   ├── prepare_nhanes.py         builds the training table
│   ├── train_model.py            trains and evaluates the detectors
│   ├── make_sample_reports.py    generates the sample reports
│   └── make_diagrams.py          architecture and data-flow diagrams
├── models/                       trained detectors + metadata
├── sample_reports/               generated samples, no real patient data
├── reports/                      evaluation output
├── docs/                         diagrams, provenance, test results
└── tests/                        80 tests
```

---

## Data sources

Everything medical in this project traces to one of three sources.

| What | Source | Licence / terms | Used for |
|---|---|---|---|
| Reference intervals | ABIM Laboratory Test Reference Ranges, January 2026 ([PDF](https://www.abim.org/Media/bfijryql/laboratory-reference-ranges.pdf)) | Published by the American Board of Internal Medicine for examination use; reproduced here as factual reference data with attribution | `data/reference_ranges.csv` |
| Explanations | [MedlinePlus Medical Tests](https://medlineplus.gov/lab-tests/), US National Library of Medicine | Public domain, attribution requested. Content MedlinePlus marks as copyrighted — the A.D.A.M. Medical Encyclopedia and the drug monographs — is **not** used | `knowledge_base/`, `data/association_rules.csv` |
| Model training data | [NHANES 2017–2018](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017), CDC / NCHS | US Government public-use data | `models/`, `data/unit_conversions.csv` |

*Source: MedlinePlus, National Library of Medicine.*

Full detail, including every NHANES variable used and the exact selection rules,
is in [`docs/dataset_provenance.md`](docs/dataset_provenance.md). A test
(`test_no_copyrighted_encyclopedia_pages_are_used`) fails the build if any
knowledge-base document comes from a copyrighted MedlinePlus section.

### Unit conversion factors

Rather than writing conversion factors from memory, they are **derived from the
data**. NHANES reports several analytes in both conventional and SI units, so
the factor is the median ratio between the two columns across thousands of
participants:

| Parameter | From | To | Factor | Derived from |
|---|---|---|---:|---:|
| Creatinine | mg/dL | µmol/L | 88.4 | 5,903 pairs |
| BUN | mg/dL | mmol/L | 0.357 | 5,901 pairs |
| Total bilirubin | mg/dL | µmol/L | 17.1 | 5,902 pairs |
| Total cholesterol | mg/dL | mmol/L | 0.02586 | 6,738 pairs |
| Folate | ng/mL | nmol/L | 2.265 | 4,571 pairs |

…and eight more. If a report uses a unit with no derived factor, the value is
**not** converted: it is shown as it was read, with a note saying why.

---

## Machine learning

### What it is for

Comparing each value with its own reference interval already catches a value
that is out of range. The model exists for the case that cannot catch: a set of
values that are each individually ordinary but rarely occur *together*.

It is not disease prediction, and the score is not a probability of illness.

### Training data

5,154 adults from the NHANES 2017–2018 public laboratory files, with 24
parameters each. Only parameters present in at least 98% of rows are used, so
there is almost no imputation during training. Strongly right-skewed columns are
log transformed — the white cell count has a skew above 50 — before scaling.

### Two detectors, and why

The brief specified Isolation Forest. It was built, evaluated, and found to be
near-blind to the one case the component exists for, so a second detector was
added rather than shipping a result that does not work.

Isolation Forest splits one parameter at a time, so it describes each
parameter's own distribution. It cannot represent the relationship *between*
parameters, so it cannot see that two values disagree with each other while both
remain individually ordinary. No amount of tuning changed this: `max_features`,
`max_samples` and the feature count were all swept, and detection stayed within
three points of the false-alarm rate throughout.

The robust Mahalanobis distance (minimum covariance determinant) uses the
covariance between parameters, so a broken relationship moves it immediately.

Both are trained and both are scored; either can raise the flag.

### Evaluation

The data has no anomaly labels, so detection is measured against faults injected
into held-out rows that the model never saw.

- **One wrong value:** a parameter is set to the training 99th percentile × m,
  or the 1st percentile ÷ m. At m = 1 the value is one that 1% of people
  genuinely have, so detection should be modest and should rise with m.
- **Inconsistent pair:** two parameters that move together (hemoglobin and
  hematocrit, MCH and MCV, iron and transferrin saturation) have their values
  swapped between rows from opposite tails. Both numbers stay real measured
  values and each is individually plausible — only the relationship breaks.

3,865 training rows, 1,289 held-out rows, flagging threshold at the 95th
percentile of training scores.

| Test | Isolation Forest | Mahalanobis |
|---|---:|---:|
| **False alarms** (untouched held-out rows) | 4.3% | 4.6% |
| One wrong value, m = 1 | 6.1% | 30.8% |
| One wrong value, m = 2 | 7.5% | 55.9% |
| One wrong value, m = 5 | 7.4% | 69.4% |
| **Inconsistent pair** | 5.3% | **77.2%** |

Across five random seeds the held-out flag rate varies by 0.07 percentage points
for Mahalanobis and 0.39 for Isolation Forest.

Reading the table: Isolation Forest sits within about two points of its own
false-alarm rate on every fault, which is to say it detects nothing. Mahalanobis
catches three quarters of broken correlations at the same false-alarm rate.

Full numbers in [`reports/ml_evaluation.json`](reports/ml_evaluation.json);
chart in `reports/ml_evaluation.png`.

### Limitations of the model

- A US survey population measured on that survey's analysers. "Unusual" means
  unusual relative to that group, not clinically abnormal.
- NHANES survey weights are not applied, so the sample is not nationally
  representative as used here.
- No thyroid or vitamin B12 data exists in this NHANES cycle, so those
  parameters cannot take part in the combination check.
- Parameters missing from an uploaded report are filled with the population
  median, which pulls the result towards "usual". The application states how
  many were missing.

---

## AI and retrieval

**Retrieval.** The knowledge base holds 39 documents built from MedlinePlus
Medical Tests pages, split into sections (what the test measures, low results,
high results, other factors). Retrieval uses Sentence Transformers with a FAISS
index when both are installed, and scikit-learn TF-IDF with cosine similarity
otherwise. The TF-IDF path is a complete retriever, not a stub, and the
interface shows which one ran.

**Explanations.** The rule-based explainer is the default and always available.
It needs no API key and no network, and everything it says traces to a row in
`association_rules.csv` or a retrieved passage, both of which carry source URLs.

**The LLM layer is optional.** It receives only the structured findings and the
retrieved passages — never a name, and nothing that identifies anyone — and is
instructed to stay inside them. Its output is then checked against a pattern
that rejects diagnoses, medicine names and doses; rejected output is discarded
and the rule-based text is shown instead. A missing key, a timeout or an API
error produces the same fallback, with a note on screen.

Three tests cover this: a provider that fails, a provider that returns unsafe
text, and a provider that returns acceptable text.

---

## Testing

```bash
python -m pytest -v
```

80 tests, running in about 5 seconds, with no network access and no API key.
They cover PDF extraction, OCR, alias matching, unit normalisation, reference
comparison (including a report interval overriding the stored table), missing
values, implausible values, the review round-trip, association rules, the
knowledge base, retrieval, model execution, LLM failure and rejection, PDF
generation, and the display helpers.

Saved output: [`docs/test_results.txt`](docs/test_results.txt).

A few tests are worth singling out:

- `test_report_interval_overrides_the_stored_table` — the same hemoglobin value
  is normal against the report's printed interval and low against the stored
  one, proving precedence works.
- `test_unverified_parameter_gets_no_judgement` — urea returns `unknown`, not a
  guess borrowed from BUN.
- `test_plausibility_window_is_wide_enough_for_real_illness` — a hemoglobin of
  4.0 g/dL is genuinely low and must not be dismissed as an OCR error, while 148
  must be.
- `test_unsafe_llm_output_is_rejected` — "Take 65 mg of iron twice daily" never
  reaches the user.

---

## Privacy

- The patient name is **never extracted**. Only age and sex are read, because
  sex selects the reference interval and age decides whether adult intervals
  apply at all.
- Uploads are processed in memory for the session and are not written to disk.
- No real patient report is in this repository. The sample reports are generated
  by `scripts/make_sample_reports.py` with invented values and a fictional
  laboratory, and are stamped SAMPLE.
- The API key is read from an environment variable. `.env` is in `.gitignore`.
- Nothing leaves the machine unless AI rewriting is switched on, and then only
  the findings and the reference passages are sent.

---

## Screenshots

`docs/screenshots/` holds the generated PDF report pages. Screenshots of the
running Streamlit interface need to be captured on the machine where it runs —
see *Known limitations* below.

---

## Known limitations

- **The Streamlit interface has not been run.** It was built and checked
  statically: all seven pages were executed against a stand-in for Streamlit, in
  both the empty and the analysed state, and all seven ran without error. That
  verifies the application's own logic, not Streamlit's rendering or the visual
  layout. Expect to adjust spacing and wording on first run.
- The Plotly figures were likewise not rendered. The data they draw is tested;
  the drawing calls are not.
- Extraction is tuned on generated samples and common Indian and US report
  layouts. An unusual layout may need new aliases in
  `data/reference_ranges.csv`.
- eGFR is read if the report prints it, but is never calculated. The published
  equations are not reproduced here, in keeping with the rule against writing
  medical formulas from memory.
- OCR quality drives everything downstream. A sharp, straight, evenly lit image
  matters more than any setting.

## Future work

- Capture Streamlit screenshots and adjust the layout after a first run
- Add thyroid and B12 to the combination model from a NHANES cycle that has them
- Per-laboratory reference profiles, so a frequently used laboratory's own
  intervals can be stored
- Trend tracking across several reports over time
- A larger knowledge base, once more public-domain sources are checked

---

## Acknowledgements

Reference intervals from the American Board of Internal Medicine. Explanatory
content from MedlinePlus, National Library of Medicine. Training data from the
National Health and Nutrition Examination Survey, CDC / NCHS.

**BloodWise AI is an educational screening and information tool. It is not a
medical diagnostic system and cannot diagnose any condition. Discuss any result
with a qualified healthcare professional.**
