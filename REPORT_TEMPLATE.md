# Accent bias in Whisper speech recognition on English read speech

**Author:** _[your name]_
**Date:** _[date]_
**Code and data:** _[repository URL]_ · dataset: Weinberger, S. (2013). *Speech accent archive*. George Mason University.

> **How to use this template.** Every `_[...]_` placeholder is yours to fill.
> All numbers should be read off the generated tables in `results/` — do not
> round by eye or retype from memory, and do not report a number this pipeline
> did not produce. Target length is 2–3 pages excluding references.

---

## 1. Motivation

_[Why accent bias in ASR matters: where these systems sit in real workflows —
dictation, captioning, voice interfaces, clinical and legal transcription — and
what it costs a speaker when the system works less well for them. Two or three
sentences.]_

_[Why Pakistani-accented English specifically: your own interest, the size of
the speaker population, its under-representation in ASR evaluation. One or two
sentences.]_

**Research question.** _[State it as one sentence, e.g.: does whisper-large-v3
produce a higher word error rate on read English from speakers born in Pakistan
than on read English from comparison accent groups, holding the text constant?]_

**Hypothesis.** _[State the direction you expected before running anything, and
say plainly whether you formed it before or after seeing the numbers.]_

---

## 2. Data

**Source.** Speech Accent Archive (Weinberger, 2013), Kaggle mirror
`rtatman/speech-accent-archive`. Every speaker reads the same elicitation
paragraph, reproduced in `reference.txt`:

> Please call Stella. Ask her to bring these things with her from the store: Six
> spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack
> for her brother Bob. We also need a small plastic snake and a big toy frog for
> the kids. She can scoop these things into three red bags, and we will go meet
> her Wednesday at the train station.

**Accent groups.** Defined from the metadata by the `--rule` arguments given to
`src/prepare_data.py`; reproduce the exact rules you ran here:

_[paste the exact `--rule` arguments you used]_

**Sample.** Fill from `results/summary_by_accent.csv` (column `n`) and
`audio/manifest.csv`:

| Accent group | Definition | n clips | Sex (M/F) | Age range |
|---|---|---|---|---|
| pakistani | _[rule]_ | _[n]_ | _[m/f]_ | _[range]_ |
| hindi | _[rule]_ | _[n]_ | _[m/f]_ | _[range]_ |
| arabic | _[rule]_ | _[n]_ | _[m/f]_ | _[range]_ |
| english_us | _[rule]_ | _[n]_ | _[m/f]_ | _[range]_ |
| english_uk | _[rule]_ | _[n]_ | _[m/f]_ | _[range]_ |

_[Note any group that fell below 10 clips — the preparation step warns about
these — and say what that does to the strength of the claims that follow.]_

---

## 3. Method

**Model and access.** `openai/whisper-large-v3`, queried through the Hugging
Face Inference API on _[date(s) of the run]_. No local inference. Per-clip
transcripts, the model id and UTC timestamps are cached in
`results/transcripts.jsonl`.

**Normalisation.** Reference and hypothesis are passed through the same
function (`src/normalise.py`): Unicode folding, apostrophe unification,
lowercasing, removal of everything that is not a letter or an apostrophe, and
whitespace collapsing. Both sides are treated identically, so the comparison is
symmetric.

**Metrics.** Per clip, `jiwer.process_words` yields WER together with
substitution, deletion, insertion and hit counts (`src/metrics.py`).

**Wrong-language outputs.** A clip is flagged when more than 50% of the letters
in its transcript are non-Latin (Urdu, Arabic, Devanagari and similar), with a
minimum-length guard. Every headline figure is reported both over all clips and
excluding flagged clips. _[Say why this distinction matters for your reading of
the results.]_

**Statistics.** Per-accent mean, standard deviation, median, and a 95%
percentile bootstrap confidence interval for the mean (10,000 resamples, seed
12345). The Pakistani group is compared with each other group by a two-sided
Mann-Whitney U test; the family of p-values is Holm-corrected; Cliff's delta
reports the effect size. Rank-based tests were chosen because per-clip WER is
bounded, skewed and measured on small samples.

---

## 4. Results

### 4.1 Overall

_[From `results/overall_summary.csv`: mean WER over all clips, over clips
excluding flagged ones, and the number of flagged clips. One short paragraph.]_

### 4.2 By accent group

**Table 1 — WER by accent group.** Source: `results/summary_by_accent.csv`.

| Accent | n | Mean WER | SD | Median | 95% CI | n flagged | Mean WER (excl. flagged) |
|---|---|---|---|---|---|---|---|
| _[…]_ | | | | | _[low, high]_ | | |

**Figure 1 — `results/figures/wer_by_accent.png`.** Boxplot of per-clip WER by
accent with individual clips overlaid.

_[Describe what the figure shows: ordering of the groups, how much the
distributions overlap, whether any group's spread is driven by a few clips.
Resist reading a difference into overlapping intervals.]_

### 4.3 Pakistani group vs each comparison group

**Table 2 — Pairwise comparisons.** Source: `results/pairwise_tests.csv`.

| Comparison | n (a/b) | Median a | Median b | U | p | p (Holm) | Cliff's δ | Magnitude |
|---|---|---|---|---|---|---|---|---|
| pakistani vs _[…]_ | | | | | | | | |

_[State which comparisons survive Holm correction and which do not. Report the
effect sizes alongside — a significant p-value with a negligible delta and a
non-significant p-value with a large delta both need saying out loud. If nothing
survives correction, say so plainly; that is a result.]_

---

## 5. Error analysis

### 5.1 Error types

**Figure 2 — `results/figures/error_types_by_accent.png`**, with counts in
`results/error_types_by_accent.csv`.

_[Do the groups differ in the *kind* of error, not just the amount? Substitution-
heavy profiles suggest mis-heard words; deletion-heavy profiles suggest dropped
or skipped speech; insertion-heavy profiles suggest hallucinated content.]_

### 5.2 Hardest words

**Figure 3 — `results/figures/hardest_words.png`**, with the table in
`results/hard_words.csv` (full per-word rates in `results/word_error_rates.csv`).

_[Which reference words does the Pakistani group lose most often, and how do the
same words fare pooled across the other groups? Comment on any phonetic pattern
you can actually support — for example dental/retroflex contrasts, consonant
clusters, or vowel length — and flag where you are speculating. The paragraph is
69 words long, so per-word rates rest on few observations; say so.]_

### 5.3 Wrong-language outputs

**Source:** `results/wrong_language_counts.csv`.

_[How many clips per accent came back in a non-Latin script, and which scripts.
This is a distinct failure mode: the model chose a language rather than
mis-hearing words. Note how much of the group's WER gap it accounts for.]_

---

## 6. Limitations

_[Adapt from the README's limitations section and keep the ones that actually
bear on your claims. At minimum cover:]_

- Read speech only, one fixed passage — no conversational speech.
- Small per-group samples; wide intervals and limited power.
- A single model queried at one point in time through a hosted API.
- Accent groups are proxies built from country of birth and self-reported native
  language, not phonetic assessment; within-group variation is large.
- Speaker-level confounds (age, sex, age of English onset, recording conditions)
  are not controlled.
- WER is a proxy for usefulness, not a measure of harm.
- _[Anything specific to your run: an under-sized group, failed clips, API
  errors, or a group whose rule matched fewer rows than expected.]_

---

## 7. Ethical considerations

_[Cover, in your own words:]_

- **Framing.** Higher error rates are a property of the model, not a deficiency
  in the speakers. Say this explicitly; the framing matters.
- **Labels.** Accent group labels describe recordings and metadata, not people's
  identities or intelligibility, and national labels flatten real linguistic
  diversity.
- **Consent and terms.** The archive's recordings were collected for linguistic
  research; the audio is not redistributed with this code.
- **Downstream stakes.** _[Where ASR errors of this size would land in practice —
  captioning, clinical notes, voice interfaces, automated interviews — and who
  absorbs the cost.]_
- **What this study cannot license.** _[No claim about the speakers, about any
  individual, or about other models or domains.]_

---

## 8. Conclusion

_[Three to five sentences: the answer to the research question, with the
direction and rough size of any difference and its uncertainty; the most
interesting thing in the error analysis; and one concrete next step — more
clips, spontaneous speech, a second model, or per-speaker modelling of the
confounds. Do not overstate: if the evidence is weak, say the evidence is
weak.]_

---

## References

Weinberger, S. (2013). *Speech accent archive*. George Mason University.
<http://accent.gmu.edu>

_[Add: the Whisper model or paper; jiwer; any accent-bias or ASR-fairness
literature you cite; the Holm (1979) and Cliff (1993) methods if you cite them
formally.]_

---

## Appendix — exact commands run

_[Paste the exact `prepare_data` and `run` commands you executed, plus the date
of the transcription run, so the tables can be regenerated.]_
