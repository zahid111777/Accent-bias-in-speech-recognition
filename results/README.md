# ⚠️ Partial run — do not interpret these numbers

The tables and figures in this folder come from an **incomplete transcription
run** and must not be read as results of the study.

## What happened

The run of 2026-09-20 stopped when the Hugging Face Inference API returned
`402 Payment Required` — the account's monthly included credits were exhausted.
Of 129 selected clips, **27 were transcribed and 102 failed**.

Because clips were processed in accent order at the time, the surviving sample
is badly unbalanced:

| accent | clips transcribed | clips selected |
|---|---|---|
| arabic | 11 | 30 |
| english_uk | 9 | 30 |
| hindi | 3 | 18 |
| english_us | 2 | 30 |
| **pakistani** (focus group) | **2** | **21** |

With `n = 2` in the focus group and `n = 2` in one comparison group, the
per-accent means, the bootstrap intervals, the Mann-Whitney tests and the
word-level rankings carry no information about accent bias. They are committed
only so the pipeline's output format is visible in the repository.

Two specific numbers in `pairwise_tests.csv` are worth calling out so they are
not quoted out of context. The Cliff's delta of `1.00` for pakistani vs
english_us is forced arithmetically by comparing two observations against two,
not evidence of a large effect. And the Pakistani mean WER currently sits
*below* the Arabic mean, which reverses the study's hypothesis — that is an
artifact of which clips were reached before the credits ran out, not a finding.

## Sample sizes, stated plainly

Even after a complete run, two groups cannot reach 30 clips, because that is
all the Speech Accent Archive contains under these definitions:

| accent | rule | available | target |
|---|---|---|---|
| arabic | `native_language=arabic` | 102 | 30 (capped) |
| english_us | `country=usa & native_language=english` | 373 | 30 (capped) |
| english_uk | `country=uk & native_language=english` | 65 | 30 (capped) |
| **pakistani** | `country=pakistan` | **21** | 21 (all) |
| **hindi** | `native_language=hindi` | **18** | 18 (all) |

So the study is capped at **21 clips for the focus group**, against 30 for each
capped comparison group. That is a small sample for the focus group, and the
confidence intervals should be expected to be wide even when the run completes.

The Pakistani group is also **not homogeneous in first language**: the 21
speakers comprise urdu 12, punjabi 3, pashto 2, and one each of english,
french, hindko and sindhi. "Pakistani" here means country of birth as recorded
in the archive's metadata, not a single accent.

## What was fixed as a result

Clip ordering is now round-robin across accents (`src/transcribe.py`), so a run
that stops early leaves a balanced sample, and a resumed run favours the groups
furthest behind. A fatal API status now aborts the run immediately instead of
repeating the same error for every remaining clip.

## Reproducing a valid run

The 27 transcripts already paid for are cached in `results/transcripts.jsonl`
(git-ignored), so resuming only pays for what is missing:

```bat
python -m src.run --audio_dir audio --out results
```

Replace this file with a description of the completed run — date, model, and
the final per-accent counts — once every clip has a transcript.
