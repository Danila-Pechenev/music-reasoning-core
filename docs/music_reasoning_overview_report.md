# Music Reasoning Task Generators

## 1. Specification Overview

The current music reasoning specification describes **8 task families**:

1. `pitch_interval_reasoning`
2. `chord_roman_reasoning`
3. `key_scale_mode_reasoning`
4. `rhythm_meter_reasoning`
5. `harmony_progression_reasoning`
6. `voice_leading_reasoning`
7. `formal_music_transformations`
8. `analysis_representation_reasoning`

So far, **2 families are implemented**.

| Implemented family | Short description |
|---|---|
| `pitch_interval_reasoning` | Generates pitch, interval, transposition, interval-size comparison, pitch-class counting, and key-context interval-classification tasks. It focuses on exact spelling, octave/register handling, interval quality/number logic, and notation interpretation. |
| `chord_roman_reasoning` | Generates chord quality, inversion, voicing, enharmonic chord equivalence, chromatic chord-label, diatonic membership, Roman-numeral analysis, and chord-realization tasks. It focuses on tertian chord structure, chord-tone order, figured bass, key context, and Roman-numeral function. |

## 2. Mode-Level Diversity Estimates

The following estimates use **difficulty level 5**. At this level:

- `pitch_interval_reasoning`: `chain_len = 4`, `max_interval_number` can reach `16`, `n_candidates = 9`, `max_accidental = 2`, `key_complexity = 7`, `p_abc = 0.5`.
- `chord_roman_reasoning`: `p_seventh = 0.6`, `max_accidental = 2`, `key_complexity = 7`, `p_secondary = 0.5`, `p_abc = 0.5`.

For these music generators, difficulty level should be understood primarily as a **dataset-distribution parameter**. Higher levels broaden the sampling space and increase the probability of harder musical features, while preserving simpler cases. Thus, a level-5 dataset may still contain tasks that look easy in isolation; the level describes the overall generated distribution rather than a strict set of tasks of exactly difficulty 5. This is an important difference from many other task generators, and it comes from the peculiarities of musical reasoning.

Counting convention:

- Counted as different: notation family, ABC key `K:`, note/chord spellings, octave numbers, interval qualities/numbers, key/mode contexts, instrument choices, operation chains, and random order of note/chord collections when that order affects the reasoning task.
- Not counted as different: alternative prompt openers/tails, `L:` values, `M:` values, and ABC duration suffixes.
- Therefore, the table estimates the number of **different underlying tasks**. The actual number of possible **surface formulations** is many times larger because `L:`, `M:`, duration suffixes, opener/tail wording, and answer-choice ordering are also randomized.
- Values are rounded order-of-magnitude estimates. Exact closed forms are messy because several modes use rejection sampling to stay within supported spelling and accidental bounds.

### `pitch_interval_reasoning`

| Mode | Approx. different tasks at level 5 | Main diversity sources |
|---|---:|---|
| `interval_naming` | ~125,000+ | start note, constructed end note, interval number up to 16, double accidentals, SPN/compact ABC/full ABC, ABC key signatures |
| `interval_arithmetic` | ~1,000,000,000+ | four-step chains; add/subtract/reduce-then-invert operations; interval qualities and compound interval numbers |
| `pitch_count` | ~1,000,000,000,000,000,000,000+ | 9-note lists, pitch-class subsets, enharmonic spellings, octaves, order of notes, notation family and ABC key |
| `interval_classification` | ~50,000-100,000 | major/natural-minor/harmonic-minor/melodic-minor contexts, all conventional key signatures, diatonic vs chromatic note pairs |
| `interval_size_comparison` | ~800,000,000+ | two written intervals, equal/different exact semitone spans, interval qualities and compound numbers, octaves, notation family and ABC key |
| `instrument_transposition` | ~24,000+ | 12 common transposing instruments, written pitch spellings/octaves, SPN/compact ABC/full ABC, ABC key signatures |
| `interval_construction` | ~500,000+ | start note, interval quality/number, above/below direction, answer-with-octave vs no-octave cases, notation family and ABC key |
| `transposition_chain` | ~150,000,000,000+ | four transposition steps, each with direction and interval, accidental constraints, start note, notation family and ABC key |

### `chord_roman_reasoning`

| Mode | Approx. different tasks at level 5 | Main diversity sources |
|---|---:|---|
| `chord_quality` | ~100,000+ | triads and seventh chords, root spellings, double accidentals, random chord-tone order, SPN/compact ABC/full ABC, ABC key signatures |
| `inversion` | ~400,000+ | chord quality, root spelling, inversion/bass member, random chord-tone order, notation family and ABC key |
| `open_close_voicing` | ~500,000+ | triads/sevenths, open vs close voicing, randomized open-voicing registers, octave placement, notation family and ABC key |
| `enharmonic_chord_equivalence` | ~500,000,000+ | yes/no cases, enharmonic respellings, pitch-class sets, optional octave-bearing chords, random order, notation family and ABC key |
| `chromatic_chord_label` | ~50,000+ | 5 chromatic chord labels, major/minor keys, key complexity, with/without octave, ordered vs unordered chord prompts, notation family and ABC key |
| `chord_membership` | ~2,000,000+ | major/natural-minor keys, diatonic and altered non-diatonic chords, triads/sevenths, optional octaves, random order, notation family and ABC key |
| `roman_numeral_from_chord` | 1,000,000+ | major/minor keys, diatonic and secondary Roman figures, triad/seventh inversions, optional octaves, random order, notation family and ABC key |
| `chord_from_roman_numeral` | ~10,000+ | key, Roman figure, triad/seventh quality, inversion suffix, secondary functions, SPN vs compact ABC answer policy |

## 3. Mechanisms Used To Make Tasks Diverse

### Different notations

The generators use three notation styles:

| Notation | Example |
|---|---|
| Scientific pitch notation, SPN | `F#3`, `Bbb1`, `C##4-G##4-B#4-E#5` |
| Compact ABC note tokens | `"^F"`, `"_B,"`, `"=E"`, `"^^c'"` |
| Full ABC score fragments | <code>L:1/8<br>M:2/4<br>K:Eb<br>&nbsp;&nbsp;[^A=BD^F] &#124;] %1</code> |

Full ABC examples are especially useful because `K:` can change how note tokens are interpreted when accidentals are omitted. CoTs explicitly resolve this when relevant, for example, “The key signature `K:Fm` makes B flat, so the score note `B` represents `"_B"`.”

### Varying prompt openers and tails

The same underlying task can be asked through several precise surface forms.

Examples from one `pitch_interval_reasoning` mode, `interval_naming`.

Openers:

- `Name the interval from {start} to {end}.`
- `Identify the interval from {start} up to {end}.`
- `What interval goes from {start} to {end}?`
- `Classify the interval formed by {start} and {end}.`

Tails for the same mode:

- `The answer is one interval name, including compound/simple size as written.`
- `Give one interval name, preserving the written interval number rather than reducing compound intervals to simple ones.`
- `Answer with one interval name, preserving the written simple or compound interval number.`
- `The expected answer is one interval name with the written interval number preserved.`

Examples from one `chord_roman_reasoning` mode, `roman_numeral_from_chord`.

Openers:

- `In {key}, treat {chord} as a chord-tone collection with {bass} in the bass.`
- `Give the Roman numeral for {chord} as a chord-tone collection with {bass} in the bass in {key}.`
- `Analyze {chord} as chord tones over bass {bass} in {key}.`
- `Find the compact Roman numeral for {chord} as a chord-tone collection with bass note {bass} in {key}.`

Tails for the same mode:

- `The answer is one compact Roman numeral with figured-bass digits closed up.`
- `Give one compact Roman numeral, closing up figured-bass digits.`
- `The expected answer is one compact Roman numeral with closed-up figured bass, for example, V65/V.`
- `Answer with one compact Roman numeral and closed-up figured bass.`

These variants are not counted in the diversity table above, but they increase the number of visible prompt formulations.

### Randomization Beyond Core Music Data

The generators also randomize:

- `M:` meter in ABC scores, e.g. `M:2/4`, `M:4/4`, `M:6/8`, `M:none`;
- `L:` default note length, e.g. `L:1/4`, `L:1/8`, `L:1/16`;
- `K:` key signatures, including major and minor keys;
- ABC duration suffixes, including sometimes omitting them;
- octave numbers where they affect the task;
- note/chord spellings, including double sharps and double flats;
- chord-tone order for unordered chord-tone collections;
- answer-choice ordering for label tasks;
- operation chains in interval arithmetic and transposition-chain tasks.

## 4. Tools And Verification Strategy

### `pitch_interval_reasoning`

The core pitch/interval solver is implemented manually inside the project. This was intentional because the generator needs detailed, controllable reasoning traces: interval number, semitone distance, major/perfect reference form, quality inference, inversion rules, transposition chains, and explicit ABC key-signature resolution. A third-party library could compute answers, but it would not directly produce the exact pedagogical CoT steps we want.

`music21` is still used in tests to verify important behavior independently, especially interval semitone sizes, interval construction/transposition, and pitch spelling compatibility.

### `chord_roman_reasoning`

This family uses both project-owned logic and `music21`.

`music21` is used where it is strongest and reliable:

- constructing and validating Roman numerals in major/minor keys;
- deriving chord pitches from Roman figures;
- checking chord qualities/common names;
- validating chromatic sonorities such as Neapolitan and augmented-sixth chords;
- providing an external reference for tests.

Project-owned logic is still used for rendering, prompt diversity, answer normalization, ABC key-resolution CoTs, chord-tone ordering explanations, metadata, and dataset-specific reasoning traces.

## 5. Tests And Coverage

Command run:

```bash
python -m coverage erase
python -m coverage run -m pytest tests/test_music_theory.py tests/test_pitch_interval_reasoning.py tests/test_chord_roman_reasoning.py
python -m coverage report src/music_reasoning_tasks/_music_theory.py src/music_reasoning_tasks/pitch_interval_reasoning.py src/music_reasoning_tasks/chord_roman_reasoning.py
```

Result: **94 passed**.

| Test file | Collected tests | Main target | Covered source file | Coverage |
|---|---:|---|---|---:|
| `tests/test_music_theory.py` | 43 | shared note, interval, scale, ABC, rendering, normalization, and music21 adapter logic | `src/music_reasoning_tasks/_music_theory.py` | 94% |
| `tests/test_pitch_interval_reasoning.py` | 22 | pitch/interval generator modes, metadata reconstruction, scoring, ABC rendering, interval arithmetic, transposition chains | `src/music_reasoning_tasks/pitch_interval_reasoning.py` | 93% |
| `tests/test_chord_roman_reasoning.py` | 29 | chord/Roman generator modes, scoring, music21 cross-checks, Roman pools, key complexity, ABC prompts | `src/music_reasoning_tasks/chord_roman_reasoning.py` | 97% |
| **Total** | **94** | all implemented music reasoning code | three music modules together | **95%** |

## 6. Curated Examples

The full curated level-5 showcases are stored as separate files in this repository rather than embedded here. They were selected from generated pools to show broad mode coverage, difficult spellings, multiple notation styles, ABC key-signature resolution, interval arithmetic, Roman-numeral reasoning, chord-quality reasoning, and enharmonic equivalence.

| Family | Curated examples file | Contents |
|---|---|---|
| `pitch_interval_reasoning` | [`examples/pitch_interval_showcase_l5.txt`](../examples/pitch_interval_showcase_l5.txt) | 24 selected level-5 examples with prompt, answer, and CoT. |
| `chord_roman_reasoning` | [`examples/chord_roman_showcase_l5.txt`](../examples/chord_roman_showcase_l5.txt) | 24 selected level-5 examples with prompt, answer, and CoT. |

Use these files for presentations, demos, and quick manual inspection. Larger JSONL or readable pools should be regenerated with `scripts/generate_examples.py`.
