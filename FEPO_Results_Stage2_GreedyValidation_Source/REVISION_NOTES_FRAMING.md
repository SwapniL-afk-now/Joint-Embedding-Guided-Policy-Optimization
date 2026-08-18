# Framing revision

- Reframed the manuscript for formal conference presentation rather than implementation-first exposition.
- Simplified the title.
- Rewrote the abstract with no equations and no code-oriented wording.
- Rewrote the introduction as a five-paragraph problem -> blind spot -> gap -> method -> contribution narrative.
- Moved Related Work before Preliminaries and reduced the main-text literature inventory; extended coverage remains in Appendix A.
- Reduced Preliminaries to only the notation and group-relative objective needed by FEPO.
- Reorganized the method as: failure reference/anchor/gate -> constrained objective -> reverse-KL interpretation -> reference updates/algorithm -> compact analysis.
- Moved estimator and moving-reference technical detail out of the main narrative and into the existing appendices.
- Split Limitations and Conclusion.
- Preserved all empirical tables and figures as placeholders.
- Preserved the verified KL directions: forward actor-to-anchor and reverse failure-to-actor.

## Terminology correction
- GRPO, GSPO, DAPO, NGRPO, AVSPO, and FEPO are described as RL algorithms/methods, not parameter optimizers.
- Mathematical expressions are called objectives/surrogates.
- The word optimizer is reserved for numerical parameter optimizers such as AdamW.
