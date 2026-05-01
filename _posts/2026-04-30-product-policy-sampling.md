---
title: "Gentle Distillation: Pedagogical Learning Without Surprises"
date: 2026
categories: [LLM, RL, Distillation, Reasoning]
---

# Gentle Distillation: Pedagogical Learning Without Surprises

*A good teacher does not just know the answer — they know how to explain it in a way the student can follow.*

**Authors:** 

---

## 1. On-policy distillation: the appeal and the wall

There is a growing consensus that keeping training *on-policy* is the right approach for post-training reasoning models. RL-based methods and on-policy self-distillation share a common motivation: by rolling out with the student's own policy, we avoid the exposure bias that plagues SFT — the mismatch between clean training prefixes and the messy, error-prone prefixes the model conditions on at test time.

On-policy self-distillation goes further by replacing the sparse final-answer reward with dense token-level supervision. A privileged teacher — one with access to a verified solution or execution trace — provides a next-token distribution at every step, and the student is trained to match it along its own rollouts:

$$
\mathcal{L}(\theta) = \mathbb{E}_{y \sim \pi_\theta}\!\left[\sum_{t=1}^{T} \mathrm{KL}\!\bigl(\pi_\theta(\cdot \mid y_{<t}, x) \;\|\; \pi_T(\cdot \mid y_{<t}, x, c)\bigr)\right].
$$

This is appealing in theory. In hard settings — long-horizon math, multi-step code generation — it quietly breaks. The culprit is a hidden assumption: that the teacher remains a *pedagogically useful* signal at every prefix the student might generate. It doesn't.

**Key Failure of OPD** : Once the student makes an early mistake, the teacher is forced to condition on a corrupted prefix. It is no longer teaching from a clean reasoning state — it is responding to a context that contradicts the privileged solution it holds. Its next-token distribution may become high-entropy, weakly corrective, or semantically inconsistent.

**[FIGURE: Teacher signal degrades when conditioned on mistake trajectories]**


> *Key message: Pure on-policy self-distillation becomes pedagogically unreliable when the teacher must respond to mistake-conditioned prefixes rather than neutral ones.*

We can measure this collapse directly. As the student generates more tokens before the teacher is queried, teacher entropy rises and recovery accuracy drops sharply:

**[FIGURE 1a: Teacher recovery accuracy vs. prefix length — drops from 73% to 39%]**

**[FIGURE 1b: Teacher entropy vs. prefix length — entropy explodes and approaches uniform (max entropy)]**

The pedagogical picture is clear: if pass@k is low and the student rarely finds correct trajectories, on-policy distillation spends most of its budget supervising from a teacher that is no longer teaching anything useful.

---

## 2. The off-policy alternative — and the same root failure

The natural response is to avoid student rollouts entirely: train on gold-label solutions, human demonstrations, or teacher-generated completions. Off-policy methods guarantee high-quality supervision — the teacher is always conditioning on clean, correct trajectories.

But they introduce a symmetric failure. Good pedagogy requires meeting the student where they are. Training on trajectories the student finds highly improbable — demonstrations requiring reasoning steps far from the student's current ability — produces gradient signal the student cannot follow. The learning target is *too surprising*.

| Approach | Signal quality | Learnable for student? | Key failure |
|---|---|---|---|
| On-policy distillation | Collapses when pass@k is low | ✓ | Teacher collapse on corrupted prefixes |
| Off-policy / SFT on gold | High | ✗ | Distributional shift; student cannot follow |
| **Gentle Distillation (ours)** | **High** | **✓** | — |

Both failures share a single underlying cause that is systematically overlooked: **surprisal mismatch**. On-policy distillation fails when the teacher is surprised by the student's context. Off-policy learning fails when the student is surprised by the teacher's tokens. The direction is opposite; the mechanism is the same.

---

## 3. Surprisal: a measure of pedagogical learnability

We define the student's *surprisal* at a response $y = (y_1, \dots, y_T)$ as

$$
S(y) \;=\; -\log \pi_\theta(y \mid x) \;=\; -\sum_{t=1}^T \log \pi_\theta(y_t \mid y_{<t}, x).
$$

This is the student's negative log-probability of the *entire response* — a direct measure of how unexpected the trajectory is. Low $S(y)$ means the trajectory is well within the student's current capability; the student can assimilate it. High $S(y)$ means the trajectory lies outside the student's distribution; gradient signal from it is noisy and potentially destabilizing.

A pedagogically useful trajectory has two properties simultaneously: (1) *high reward* — the teacher endorses it as a good reasoning path, and (2) *low surprisal* — the student can actually learn from it. We want a training distribution that controls both.

### Optimal trajectory distribution

We seek $q^*$ that stays close to both the student and the teacher at the response level. This is the solution to the weighted KL problem:

$$
q^* = \arg\min_q \;\mathrm{KL}(q \,\|\, \pi_\theta) + \alpha\,\mathrm{KL}(q \,\|\, \pi_T).
$$

Solving this yields a geometric mixture at the sequence (response) level:

$$
q^*(y \mid x) \;\propto\; \pi_\theta(y \mid x)^{\,\delta} \cdot \pi_T(y \mid x, c)^{\,1-\delta}, \qquad \delta = \frac{1}{1+\alpha}.
$$

Notice that $-\log q^*(y) \propto \delta \cdot S_\theta(y) + (1-\delta) \cdot S_T(y)$: the optimal distribution jointly minimizes student and teacher surprisal. When $\delta \to 1$: fully on-policy. When $\delta \to 0$: fully off-policy. Intermediate $\delta$ gives the pedagogically useful middle ground.

---

## 4. The generation problem

The distribution $q^*$ is the right target, but it is defined at the *sequence level*. Generating from it via standard autoregressive sampling is not straightforward.

The sequence-level product $q^*(y) \propto \pi_\theta(y)^\delta \cdot \pi_T(y)^{1-\delta}$ does factorize token-by-token, but only if we compute the product at each step:

$$
q^*(y_t \mid y_{<t}, x) \;\propto\; \pi_\theta(y_t \mid y_{<t}, x)^{\delta} \cdot \pi_T(y_t \mid y_{<t}, x, c)^{1-\delta}.
$$

This is **mix sampling**: at each token, interpolate the logits of both models, renormalize, and sample. It exactly instantiates the per-token marginal of $q^*$.

> **Mix sampling: exact but expensive.** At each token, mix sampling requires a forward pass through *both* $\pi_\theta$ and $\pi_T$ simultaneously. This doubles inference cost during rollout generation and — critically — is incompatible with optimized single-model samplers like vLLM and SGLang. It cannot scale to the batch sizes needed for practical RL training.

We need an approach that achieves the distributional goal of $q^*$ without the cost of joint decoding at every step.

---

## 5. Gentle Distillation: a pedagogical training framework

The key insight is this: mix sampling is expensive because we are combining a *fixed* teacher with a fixed student at every step. But if we instead *adjust the teacher itself* to generate low-surprisal trajectories — to teach at the student's level — we recover the same distributional target with a single forward pass.

This is the core idea of **Gentle Distillation**. Rather than mixing two policies at inference time, we train the teacher to naturally produce pedagogically appropriate content: high-reward trajectories that the student finds plausible.

### 5.1 Tilting the teacher

We fine-tune $\pi_T$ with a joint objective that rewards both task performance and low student surprisal:

$$
\mathcal{J}(\pi_T) \;=\; \mathbb{E}_{y \sim \pi_T}\!\bigl[R(y, x)\bigr] \;-\; \lambda \underbrace{\mathbb{E}_{y \sim \pi_T}\!\bigl[-\log \pi_\theta(y \mid x)\bigr]}_{\text{expected student surprisal } S(y)}.
$$

The penalty term is exactly the response-level surprisal $S(y)$ from Section 3 — not an ad-hoc regularizer, but the natural quantity that controls learnability. Minimizing it pushes the teacher toward generating tokens the student expects.

The optimal tilted teacher under this objective is:

$$
\tilde\pi_T(y \mid x) \;\propto\; \pi_T(y \mid x, c) \cdot \pi_\theta(y \mid x)^{\lambda}.
$$

> **Solving the generation problem.** The optimal $\tilde\pi_T$ is the same geometric mixture as $q^*$ (with $\delta = \lambda/(1+\lambda)$). Sampling from $\tilde\pi_T$ achieves exactly what mix sampling achieves — but requires only a single forward pass through the tilted teacher. No joint decoding. Compatible with vLLM, SGLang, and any standard sampler.

### 5.2 Why alternation is necessary — and solves the problem

Tilting the teacher once against the *current* student is not enough. As the student improves during training, $\pi_\theta$ shifts. What was low-surprisal for an earlier student may no longer be appropriate for the updated one. A static tilted teacher becomes miscalibrated.

The alternating algorithm directly resolves this: it keeps the teacher perpetually calibrated to the student's current distribution. Each tilt step recomputes the geometric product $\tilde\pi_T^{(i)} \propto \pi_T \cdot (\pi_\theta^{(i)})^\lambda$ for the latest student. The student step then trains on trajectories that are appropriately learnable for that student.

This is the sense in which alternation solves the sequence-level product problem from Section 4: rather than materializing the product at each decoding step (expensive), we bake it into the teacher's weights and refresh it between training rounds.

**Algorithm: Gentle Distillation — alternating procedure**

```
For i = 1, 2, 3, ...

  [Tilt step — Teacher]
  Given current student π_θ^(i), fine-tune π_T to produce π̃_T^(i)
  that maximizes reward while minimizing E_{y ~ π̃_T}[S_{θ^(i)}(y)].
  Optimal solution: π̃_T^(i) ∝ π_T · (π_θ^(i))^λ

  [Student step — Student]
  Sample trajectories from π̃_T^(i) using any standard sampler (vLLM, etc.).
  Train π_θ^(i+1) on these via RL or DPO,
  with per-token surprisal weighting  w_t = π_θ(y_t)^λ  in the loss.

  Set i ← i + 1.
  The updated student shifts the surprisal landscape;
  the next tilt step recalibrates the teacher accordingly.
```

### 5.3 Surprisal-weighted loss

Even with a tilted teacher, some tokens in sampled trajectories may still be more surprising to the student than others. To prevent high-surprisal tokens from dominating gradient updates, each token's loss contribution is weighted by:

$$
w_t \;=\; e^{-\lambda \cdot s_t} \;=\; \pi_\theta(y_t \mid y_{<t}, x)^{\lambda}, \qquad s_t = -\log\pi_\theta(y_t\mid y_{<t},x).
$$

This is the per-token version of the same surprisal measure from Section 3. Together with tilted-teacher sampling, surprisal is controlled at both stages: in what trajectories are generated, and in how each token contributes to training.

---

## 6. Results

We evaluate on hard mathematical reasoning benchmarks (baseline ~38% accuracy). GDRL-SurprisalBON is our full method — tilted-teacher sampling with surprisal-weighted RL. GDRL-SFT applies tilted sampling but with uniform loss weighting. GRPO is the standard on-policy RL baseline.

**[FIGURE 2: Accuracy vs. training samples — GDRL-SurprisalBON peaks at ~48.5% at 4K samples; GRPO peaks at ~43% at 8K; GDRL-SFT plateaus ~40.5% with instability after 4K; Baseline 38%]**

| Method | Peak accuracy | Samples to peak | vs. baseline |
|---|---|---|---|
| Baseline | 38% | — | — |
| GDRL-SFT | ~40.5% | ~4K | +2.5pp |
| GRPO | ~43% | ~8K | +5pp |
| **GDRL-SurprisalBON (ours)** | **~48.5%** | **~4K** | **+10.5pp** |

GDRL-SurprisalBON reaches ~48.5% using 4K samples — over 5 points above GRPO at twice the data budget. GDRL-SFT's instability after 4K confirms that the surprisal-weighted loss is not optional: tilted-teacher trajectories without down-weighting high-surprisal tokens allow a small fraction of them to destabilize training.

---

## Summary

On-policy distillation and off-policy learning are not separate problems — they are the same surprisal mismatch viewed from two directions. A pedagogically useful training signal must be both high-reward and low-surprisal for the current student. Neither standard paradigm guarantees this.

Gentle Distillation addresses it at the root. The optimal training distribution is a geometric mixture of student and teacher at the response level, but sampling from it exactly via mix sampling is incompatible with efficient inference. By tilting the teacher to natively produce low-surprisal, high-reward trajectories, and alternating this tilting with student updates, we keep teacher and student mutually calibrated throughout training without joint decoding. A surprisal-weighted loss ensures residual high-surprisal tokens do not dominate optimization.

> **Core principle:** Good pedagogy meets the student where they are. Surprisal — the student's negative log-probability of a training trajectory — is a direct, computable measure of whether training data is within reach. Control it in both the trajectories you generate and the loss you optimize, and learning remains stable even when the task is hard and the student rarely succeeds on its own.
