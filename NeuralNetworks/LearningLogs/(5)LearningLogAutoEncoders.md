# What I learned coding Autoencoders

## Before I Start:

I have never seen autoencoders before, but they were a widespread recurring architecture while I did my research on models. It seems as though basic autoencoders shouldn't be too much of a challenge since I can simply use my MLP implementation. I want to try coding VAE's if possible since those allow for interpolation and would make for nice MNIST pictures.

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

Life got a bit in the way of the project, and it's been about a month since my last test. I will try my best to summarize the experience.

1. I successfully implemented batch-norming and autoencoders. I had good accuracy with batch-normed nets, and had some interesting results interpolating between numbers in feature space. I attempted a naive implementation of VAEs to bad results. I also re-tried convnets now with much better performance and results.
2. It was very surprising to me that you can coherently interpolate between points in an autoencoder, I did not expect non-number entries to be coherent as "something between a 6 and an 8". I can understand this in networks specifically made for generation, but I wasn't expecting it from what I saw as a "compression-only" approach.
3. I re-discovered how you can use neural networks to compress data into a latent space, and then re-use another network to de-compress from latent space to an actual image.

I think the most frustrating part about this segment was realizing that my home-cooked autograd had too many small mistakes that combined into big issues. As much as I wanted to push my autograd to its limits, I did mention in the beginning how I wanted to "abstract away" for learning purposes, and I feel like I took it farther than I thought possible. I'm not disappointed, just a little wishful that I could have implemented everything from the ground up. However, my main intention is not to learn how to use autograd but instead to learn how these architectures work and how to use them, so I will cut my losses and move on.

4. If I had infinite time, I would attempt to implement VAEs properly for better results, and attempt to generate images using a simplex VAE, as that leads to natural "extrema" images that I would like to visualize.