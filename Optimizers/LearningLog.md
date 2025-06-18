# What I learned from this section

This section is quite long, so the reflection will be longer than usual. I will write down a reflection at the end of every section to solidify learning, and from now on will try to write insights as comments as I realize I have lost a lot of the "small" details that I could've otherwise written down. Here is the outline:
### Kolb Cycle – Optimizers

1. **Concrete Experience**
    - *Snapshot:* What task or experiment did you actually run? (2–3 sentences)
    - *Outcome metric:* Final loss / accuracy / any observable result.

2. **Reflective Observation**
    - *Patterns spotted/Points of Struggle:* What surprised or puzzled you in the outputs? What was hard about this?
    - *Compare & contrast:* Link to a prior module where you saw a similar pattern.

3. **Abstract Conceptualisation**
    - *Theory bridge:* Summarize the core principle you (re)discovered—e.g., why momentum helps escape shallow minima.

4. **Active Experimentation**
    - *Next tweak:* One concrete change you would try if you had infinite time (learning rate schedule, batch norm, etc.).
    - *Prediction:* What effect do you expect and why?

---
Given that I am writing this for the entire module, I will be a little more ramble-y.
1. **Concrete Experience** I did the following:
   1. *Implemented backprop and my own autodiff engine*: 
   2. *Implemented a parameter wrapper*: Abstracted a lot of the concepts I needed so that I could just create the polynomial I wanted to optimize and pass it through a loop. This was better than having all the elements exposed.
   3. *Implemented a short training loop using backpropagation, my optimizers, and my polynomial classes*: This made training easier, as I had a really messy-looking code base before. Messier at least. It made changing hyperparameters as simple as changing a single number on my function instead of having to re-wire the entire loop every time.
   4. *Implemented SGD*: Got it to work on the polynomial problem, although it was a little slow on updating.
   5. *Implemented SGD with scheduled decay*: This let me make steps larger at the beginning. This sped up learning as I could lunge forward at the beginning without fearing overshoot at the end.
   6. *Implemented SGD with (Nesterov)Momentum*: This was one of the most successful optimizers, as the adapting speeds of learning based on gradient history proved quite useful in my toy problem.
   7. *Implemented AdaGrad/RMSProp*: Both are similar optimizers that "slow down" instead of "speeding up" like momentum-based optimizers. They converged pretty well, but not as good as Momentum
   8. *Implemented Adam*: the main goal of this section. Adam is better that AdaGrad/RMSProp but falls a little short of Momentum based methods for this toy problem.
2. **Reflective Observation** 
   1. I got it to work after much struggle. I did not understand the closure approach at first and had to resort to online resources for topological sort as I had never actually implemented the algorithm and was rusty on the concept. There was a pesky bug with exponentiation that took a surprising amount of time to debug.
   2. I thought and re-thought about how to implement parameters, as I misunderstood how backprop worked at first and thought it would be simpler to make parameters handle their own gradients and updates instead of wrapping them around Variables. In the end the wrapper is more of a simple abstraction, but still proved quite useful
   3. I originally did not intend to make this formally, which is why it is not part of another module. I quickly realized a lot of my tests were very repetitive and would gain great speedups if I could reuse my code
   4. I knew about SGD before, but it was quite surprising to actually visualize how it followed the gradient so nicely even if I knew in theory that's what it did. I was also surprised how fullbatch decreased in one dimension fast enough that updates became almost 1-D in the end, and how non-shuffled minibatch took such a regular pattern through parameter space.
   5. Scheduling decay was suprisingly effective for this toy problem. I imagine that for more complex problems not knowing how close you are to the true answer makes scheduling harder, but it was quite nice to see it "speed up" at the beginning and "slow down" at the end.
   6. I was surprised at the effectiveness of gradient accumulation in speeding up convergence. It was fun playing with the hyperparameters and watching the "ball" "roll over" and then "roll back in" when momentum was too high. Nesterov was a little hard to implement, since I assumed the optimizer did not have access to a parameter's values before backprop. I ended up going for a hardcoded approach which I am not happy with. To update velocities, I originally had a helper class which I then re-factored into the update_parameters method. This led to a strange bug where lr did not update with a lambda function which I fixed soon after.
   7. AdaGrad was a bit of a letdown at first. It seemed "too heavy" since it slowed down way too fast and didn't manage to get close to the point. I was very surprised when batched AdaGrad did so well. I hadn't fully realized that having smaller updates led to smaller gradients accumulating, and didn't expect it to fix the "too heavy" problem. RMSProp significantly helped with AdaGrad's problem and was one of, if not the fastest, converging algorithms in my tests.
   8. I will admit I was biased to liking Adam, as wanting to learn how it worked was one of the main motivators for this project. It made cool-looking pictures, I assume due to its dampening, when the lr was too high and it overshot. It was definitely very good, converging nicely regardless of where I placed the point originally or how high the lr was. However, it wasn't really able to make it as close to the best point as other algorithms, as it would start oscillating once the tolerance was set lower.
3. **Abstract Conceptualization**: Backprop is hard and I will more than gladly go back to using libraries. SGD is pretty good out of the box, and small tweaks like scheduling and momentum can make it a pretty powerful and simple scheduling algorithm. AdaGrad and RMSProp are good on their own, but adding the momentum capabilities into Adam mixed with their adaptive gradients makes for a very powerful optimizer.
4. **Active Experimentation**:
   1. If I could add a few more tweaks, I would:
      - Test my optimizers on harder problems, ideally with weight regulation so that their respective strengths can shine. I expect my issues with AdaGrad would probably be mitigated this way
      - After that, I would also implement AdamW. I would like to see convergence speed compared between both Adam and AdamW, and also prediction accuracy.
      - Add randomization to my batches so that gradients are more stochastic