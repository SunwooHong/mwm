# System 1 diagnosis

GPU job `1864790` on `cuda:0` (L40S). Unimix 0.01 KL, no free-nats floor. Paired actions: roll the frozen actor at gain 1.0 for 100 steps, then replay those actions at gain 0.7. Seeds 0, 1, 2. Memory is the 128 KiB projection. Raw file: `logdir/v2_system1_diagnosis/diagnosis.json`.

## 1. Is the gain shift visible in prior–posterior KL?

Yes. With the same actions, 100-step mean KL is about 10–18 at gain 1.0 and 54–61 at gain 0.7. The excess is roughly 40 nats, not a small additive bump on an already large baseline.

A natural actor rollout at gain 0.7 (different visited states) sits in between, about 30–37. Action magnitude and KL on the paired shift trajectory are almost uncorrelated (seed 0 correlation 0.06).

## 2. Is the local update choked by clipping, normalization, or projection?

Yes. After the fixed 64-d projection, feature norms average 0.15. The probability-delta update wants a step of size about 1.3, and the Frobenius cap `μ=2` binds on 95–99% of steps. `||P||_F` sits on that cap. The logit correction actually applied averages 0.11, against probability-error norms of about 4.

The logit-error rule is clipped to L2 0.5 on essentially every step and then also hits the same cap. Its KL drop is smaller than the probability-delta rule.

The local direction is not the bug. Autograd of the unimix KL gives cosine −0.97 with `q−p`, so adding `η(q−p)` is descent. Plain softmax and unimix agree at that sample.

## 3. Can an offline-fitted residual reduce held-out KL?

No, not under this cap. Adam on the same 128 KiB matrix, fit on the first 50 shifted transitions, changes train KL by about 0.17 (66.28 to 66.11 on seed 0) and held-out KL by about 0.08 (51.31 to 51.23). Seeds 1 and 2 match that pattern. The ceiling of this residual, not the Hebbian estimator, is the limit.

## 4. Does a better local rule beat frozen and the wrong memory?

Online probability-delta beats clipped logit-error, and both beat frozen, but only by about 0.1 KL against an excess of about 40. A memory learned on seed 0 at gain 0.7, at gain 1.0, or by the offline fit all move another seed’s shifted KL by less than 0.06. There is no gain-specific transferable correction in this store.

## 5. Does the correction change action ranking?

No in any practical sense. On 15 stored states and 4 matched candidates, the online `P` flips the argmax twice and the offline `P` never does. Rank correlation stays 0.997. Those flips are not evidence of a better action.

## 6. Decision

Do not start a closed-loop control pilot, and do not search Hebbian learning rates on this residual.

The shift is large in RSSM KL. A norm-capped 128 KiB logit residual cannot represent it, even with gradients. The next System 1 has to change the adaptation target or the capacity constraint. That is a new version, not another setting of this prior-logit rule.
