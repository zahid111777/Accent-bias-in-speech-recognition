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
