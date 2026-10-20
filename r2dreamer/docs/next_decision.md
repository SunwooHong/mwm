# Next decision

**STOP** for closed-loop control runs.

System 1 v2 lowers prequential KL on every development seed, and the no-write control matches the frozen prior. The reduction is far too small to treat as dynamics adaptation: under 0.7 KL against baselines of 22–52. A control pilot would not be interpretable as fast prediction helping behavior.

Do not launch the five-condition pilot or a 20–30 seed confirmation on this update rule and step size (`eta=0.05`, probability error added onto logits). Any revision has to change the write so that a before-update KL actually moves, and that check has to pass on these same fixed trajectories before actor experiments resume.
