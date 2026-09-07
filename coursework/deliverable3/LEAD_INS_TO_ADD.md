# Lead-in paragraphs for §1, §2 and §7 — to be pasted manually

The Notes on Research and Writing ask each section for a short introduction. These three currently
open straight into detail. Each paragraph below goes **at the very top of the file named**, before
the existing first line. Nothing else in those files changes.

**Word cost: 132 body words in total** (44 + 43 + 45). The body is at 7,885 against 8,800, and §8
plus the two red placeholders are budgeted at roughly 750, so all three fit only if §8 lands at or
under 600 words. If it runs long, §7's is the one to drop — it is the least abrupt of the three.

---

## 1 — `sections/one.tex` · §1 Patch-Based Tokenisation: a Formal Description

Currently opens on `Let $\mathbf{x} = \{x[n]\}_{n \in \mathbb{N}}$…` — the first word of the numbered
report is a quantifier.

**44 words.**

```latex
Everything that follows rests on treating patch tokenisation as an operator rather than an
implementation detail, because its degeneracies are then derivable instead of observed. This section
fixes that operator and the notation the rest of the report uses for it; the conditions under which
it loses information are derived in \autoref{sec:two}.

```

---

## 2 — `sections/two.tex` · §2 Phase Locking

Currently opens on `Consider a continuous-time periodic signal sampled uniformly at sampling
frequency $f_s$…`, then runs about 715 words of derivation before its first heading.

**43 words.**

```latex
A tokeniser can destroy information that the sampler preserved, and the condition under which it
does so is arithmetic. This section derives that condition in both of its forms, one governed by the
stride and one by the patch, and states what their union does and does not license.

```

---

## 3 — `sections/seven.tex` · §7 Results

Currently a single paragraph. This gives the section an opening that survives whatever the fits
return, so it can be written now.

**45 words.**

```latex
This section separates what the descriptive pass suggests from what the inferential stage decides.
The first is reported here only to establish that there is something to measure; the second is
reported under the rules fixed in \autoref{tab:bayesDecisions}, and returns no verdict it has not
earned.

```

---

## Also worth a lead-in, not requested

`sections/six.tex:44`, subsection *Bayesian analysis*, still carries
`% TODO: DESCRIPTION OF BAYESIAN APPROACH` and then opens on context truncation, which the
subsection has not introduced. That TODO is the missing introduction. It is the last placeholder in
the report that is not waiting on the fits.
