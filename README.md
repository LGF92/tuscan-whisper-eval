# Does Whisper Struggle with Tuscan Italian?

A small case study on accent and speech recognition, testing OpenAI's Whisper (turbo) on Tuscan-accented Italian against a non-Tuscan baseline, with linguistically motivated hypotheses.

## The question

Speech recognition systems are known to perform unevenly across accents. Tuscan Italian is an interesting test case: it has distinctive features such as the *gorgia toscana* (spirantisation of intervocalic /k t p/, e.g. *la casa* → [la ˈhasa]) and strong *raddoppiamento sintattico* (e.g. *a corte* → [akˈkorte]).

I wanted to know two things:

1. Does Whisper make more errors on Tuscan speakers than on other Italian speakers?
2. If so, can those errors be traced to specific Tuscan phonological features?

## Data

All audio comes from **Mozilla Common Voice (Scripted Speech, Italian)**, released under CC0, downloaded through Mozilla Data Collective.

**Tuscan set**

- Speakers selected through their self-reported accent label, matching Tuscan regions, cities and demonyms (e.g. *toscano*, *fiorentino*, *viareggino*, *sud senese*).
- Excluded *Tosco-Emiliano* (a border variety with weak or absent gorgia) and one label indicating an English speaker.
- Each speaker capped at 50 clips, randomly sampled with a fixed seed.
- **Result: 292 clips from 11 speakers.**

**Baseline set**

- Randomly sampled speakers with a non-Tuscan, non-foreign accent label and at least 25 clips.
- Female speakers removed, because the Tuscan set contained none, so gender couldn't confound the comparison.
- Same 50-clip cap.
- **Result: 393 clips from 9 speakers.**

**Limitations, stated up front**

- Small sample: 11 and 9 speakers.
- No female Tuscan speakers available.
- Accent labels are self-reported and free text.
- Read speech only, in which speakers may tone down their accent.

**Potential data leakage.** The comparison model, `jonatasgrosman/wav2vec2-large-xlsr-53-italian`, was fine-tuned on an earlier Common Voice Italian release, so some test clips may have been in its training data. This could make its absolute error rates look better than they really are. The Tuscan-vs-baseline gap and the within-clip feature tests (gorgia, raddoppiamento) are less affected, since any leakage applies to both sides. Whisper's training data is not public, so overlap can't be ruled out there either.

## Method

- Transcribed every clip with Whisper `turbo`, language set to Italian, on a free Google Colab GPU.
- Normalised the reference and hypothesis text: lowercase, punctuation and apostrophes removed, accents stripped, so formatting differences such as *l'ambito* and *l ambito* don't count as errors.
- Computed word error rate (WER) with `jiwer`, per group and per speaker.
- Used word-level alignment to test whether errors fall on words carrying specific phonological features.

## Results

### 1. Overall gap

| Group | WER |
| --- | --- |
| Baseline | 3.91% |
| Tuscan | 5.69% |

Tuscan speakers have about 45% more errors overall.

### 2. Per speaker

The gap is not uniform. The highest error rates come from speakers who described a strong accent (*viareggino*, *fiorentino*), while speakers reporting only a light Tuscan inflection performed at baseline level. With 11 speakers, this is a trend, not proof.

### 3. The gap is about names

Common Voice sentences come largely from Wikipedia and contain many foreign names and English words.

| Group | WER without name sentences | Share of sentences with names | WER on name sentences |
| --- | --- | --- | --- |
| Baseline | 3.01% | ~40% | 5.15% |
| Tuscan | 2.86% | ~52% | 8.39% |

On ordinary Italian sentences, the gap disappears. The difference comes from two things: Tuscan speakers happened to read more sentences containing names, and Whisper made more errors on their renditions of those names (e.g. *get lucky* → "catlaki", *Riggs* → "hicks").

### 4. Testing Tuscan phonology

I flagged words where each feature could occur and compared their error rate with all other words.

| Feature | Whisper baseline (feature / other) | Whisper Tuscan (feature / other) | wav2vec2 baseline (feature / other) | wav2vec2 Tuscan (feature / other) |
| --- | --- | --- | --- | --- |
| Gorgia toscana | 3.52% / 3.27% | 3.95% / 5.48% | 3.94% / 4.35% | 7.02% / 7.73% |
| Raddoppiamento sintattico | 3.09% / 3.34% | 5.24% / 5.24% | 3.78% / 4.32% | 7.33% / 7.63% |

Neither feature meaningfully increased either model's error rate. The largest rise is +0.25 points (Whisper, baseline, gorgia), well within noise; for Tuscan speakers, feature words were never harder than other words. Individual cases do occur, for example *a corte* → "accorte" and *fu chiuso* → "è fucciso", but they are rare.

Raddoppiamento is part of standard Italian pronunciation, so it is arguably not a Tuscan-only feature anyway. The gorgia result may reflect read speech, where speakers often reduce it, and the strength of Whisper's language model in recovering likely words from context.

### 5. A second model: wav2vec2

I repeated the analysis with `jonatasgrosman/wav2vec2-large-xlsr-53-italian`, a wav2vec2 model fine-tuned on Italian. Unlike Whisper, it has no built-in language model and transcribes more literally what it hears.

| Model | Group | WER overall | WER without names | WER on name sentences |
| --- | --- | --- | --- | --- |
| Whisper | Baseline | 3.91% | 3.01% | 5.15% |
| Whisper | Tuscan | 5.69% | 2.86% | 8.39% |
| wav2vec2 | Baseline | 4.59% | 3.32% | 6.31% |
| wav2vec2 | Tuscan | 8.10% | 5.23% | 10.85% |

The two models behave differently. For Whisper, the Tuscan gap disappears on sentences without names. For wav2vec2, it remains: Tuscan speakers get about 58% more errors even on ordinary Italian.

Neither model makes more errors on gorgia or raddoppiamento words (table above). So the accent effect seen in wav2vec2 comes from something other than these two features, such as vowel quality, rhythm or individual speakers. Whisper's larger training data and language model appear to absorb it.

## Conclusion

On this data, Whisper handles Tuscan-accented Italian well on ordinary sentences; its accent gap appears only on foreign names and English words. A more literal model, wav2vec2, shows a real accent gap even on ordinary Italian. In neither model is the gap explained by the gorgia or raddoppiamento sintattico, so the obvious hypothesis did not hold. With 11 Tuscan speakers and possible data leakage, these are leads, not proof.

## Next steps

- Test on spontaneous or conversational speech, where the gorgia is stronger.
- Add more speakers, including women, possibly from other corpora.
- Investigate which acoustic features drive wav2vec2's Tuscan gap (e.g. vowel quality, rhythm), and test a model with no Common Voice in its training data.
- Test foreign-name pronunciation directly with a controlled set of sentences.

## Reproducing this

The notebook in this repo runs end to end on a free Colab GPU. You'll need a free Mozilla Data Collective account and API key, stored as a Colab secret named `MDC_API_KEY`.
