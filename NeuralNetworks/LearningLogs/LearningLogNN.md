# What I learned re-coding Neural Nets

## Before I Start:

I honestly did not intend to re-do neural nets in the beginning. However, I realized I had the basics down well enough to extend my own autograd to a vectorized setting and implement flexible neural net architectures instead of ones with hardcoded derivatives like my first attempt. I am excited to try this out, especially since this will make convnets much easier to implement once I have NNs down.

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

1. I got a basic neural network implementation to work with a custom autodiff framework that now accepts numpy vectorized ops. I tried it on the same test problem as the basic SGD one (x^2 + x) to pretty good results. 
2. Definitely the hardest part about this was the numpy op broadcasting. I had a very hard time getting things to work, and I had not thought about how grads need to be "unbroadcasted" during the backwards call. I also had some issues with function anchoring which seems to be a common thing in python OOP according to stackexchange. This meant I had my first encounter with the __get__ method.
3. GradGraphs are hard. Much harder than I really thought they could've been. Neural networks are universal function approximators, and they can learn a variety of tasks pretty well.
4. Given that my current optimizers have no regularization, the next thing I will do is implement regularization schemes.