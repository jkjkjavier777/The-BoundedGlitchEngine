# 🔧 BoundedGlitchEngine
### Academic Essay & Study-Drafting Framework

> Deconstruct arguments. Surface friction. Draft with rigor.

---

## What This Is

A single-purpose engine, not a suite. One job: take a source text, find where it strains, and turn that strain into essay-ready material. No bot rotation, no governance zones — one process, run start to finish.

---

## Repository Layout

```plaintext
BoundedGlitchEngine/
├── README.md                  # This file
│
├── engine/
│   ├── deconstruct.md         # Argument, assumption, method extraction
│   ├── glitch-detect.md       # Paradox / gap / friction identification
│   ├── draft.md                # Outline + essay synthesis
│   └── voice.md                # Scholarly tone constraints
│
├── outputs/
│   ├── executive-summary.md   # Core material, condensed
│   ├── thesis-angles.md       # 2–3 argument hooks per text
│   ├── essay-outline.md       # Intro / body / conclusion scaffold
│   └── discussion-questions.md
│
└── input/
    └── source.md               # Drop the document here
```

---

## Operating Directives

**1. Deconstruct**
Pull assertions, assumptions, and method from the source. No summarizing what it *says* without also naming what it *assumes*.

**2. Detect glitches**
A glitch is a paradox, a logical gap, or a counter-intuitive result — not just "an interesting point." It has to be a place where the argument could be pushed and something gives.

**3. Draft**
Turn glitches into thesis angles. Turn angles into outlines. Never skip straight from summary to conclusion.

**4. Hold the voice**
Analytical, rigorous, clear. No hedging language, no filler transitions.

---

## Output Contract

Every run against a source produces exactly these four things:

| Output | Contains |
|---|---|
| **Executive Summary** | Core material, key themes — condensed, not paraphrased-at-length |
| **Thesis Angles** | 2–3 distinct, defensible arguments derived from the text |
| **Essay Outline** | Intro → thematic body (claim + evidence per paragraph) → conclusion/implications |
| **Discussion Questions** | Targeted at exam prep or seminar use |

No output ships without a corresponding glitch behind it. If nothing in the text strains, say so — don't manufacture friction.

---

## Input

```
[SOURCE MATERIAL / COLLEGE READING GOES HERE]
```

---

## Usage

Drop text in `input/source.md`. Run the four directives in order. Stop at the output contract — don't over-deliver, don't under-deliver.
