# What I learned re-coding Optimizers (Now with vectorization)

## Before I Start:

(I forgot to fill this out before I started so this is in post) I am confident that I can re-code optimizers, as I have already done it once before and now I can face this task with more experience. I am excited to implement AdamW.

### Kolb Cycle – Optimizers

1. **Concrete Experience**
    - *Snapshot:* What task or experiment did you actually run? (2–3 sentences)
    - *Outcome metric:* Final loss / accuracy / any observable result.

2. **Reflective Observation**
    - *Patterns spotted/Points of Struggle:* What surprised or puzzled you in the outputs? What was hard about this?
    - *Compare & contrast:* Link to a prior module where you saw a similar pattern.

3. **Abstract Conceptualisation**
    - *Theory bridge:* Summarize the core principle you (re)discovered

4. **Active Experimentation**
    - *Next tweak:* One concrete change you would try if you had infinite time (learning rate schedule, batch norm, etc.).
    - *Prediction:* What effect do you expect and why?

---

1. I ran a 2-layer neural network with leaky-relu (0.1 negative slope) on MNist and was able to obtain up to 95% accuracy on some preliminary tests. I then tried out all of the optimizers I have coded thus far, plus their W versions which don't accumulate regularization into the gradients. I was surprised to learn that L_infinity with regular Adam seemed to work better than AdamW with L2 for the MNist task, although I do not know if this was simply RNG.
2. This was probably one of the hardest parts to implement. First, I once again had issues with broadcasting weights. Then, I realized I had to re-factor my optimizers and neural networks to store the list of tensors directly instead of the parameter interface, since I needed to access the value of each parameter directly. After some more refactoring I got it to work in a way where I don't need to store the parameters directly anymore, but I left it as-is for now. Once the optimizers worked I wondered if my implementation of AdamW is correct, as it performed significantly worse than expected. I think I am almost out of broadcast hell, so I will keep pressing on. One of my goals was to implement transformers using my own autodiff engine, but I'm genuinely considering switching to torch or jax just to access their autodiff engines for speed and ease of mind. We'll see how things go for RNN's, and I'll decide from there.
3. I re-discovered how regularization can help a neural network achieve more generalizable results, how different kinds of regularization can lead to different parameter distributions, and how I should not take broadcasting for granted in autodiff engines.
4. Try to re-do W versions of algorithms and try different problems to see if I could get them to learn better than their non-W counterparts. I really expected AdamW to be better than Adam for MNIST, and I expect that after a few tweaks it really would be the best algorithm.