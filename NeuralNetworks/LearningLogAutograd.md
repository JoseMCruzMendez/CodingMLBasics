# What I learned re-coding Autograd

## Before I Start:

In order to make my life easier, I will remake my autograd to allow for vectorized numpy ops, at least for this section. With some experience under my belt I think it should be easier than the first time. I will reuse my optimizer code and have my new autograd expose a similar API to avoid bugs. I will also add regularizers to my optimizers, that punish NNs with weights that are too large. This means I will also try to implement AdamW, and see if I have time to implement AdaMax and a W version of AdaMax now that I read the paper by Kingma and Ba.

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

1. I finished setting up the autograd with numpy, wrapping old functions and implementing new ones like matmul, sum, avg, sigmoid, ReLU, and BCE. The grads all work and should make setting up a neural network easier than the first time I did it, where I hand-coded my derivatives as I didn't know about autograd.
2. Although having set up autograd before really did help me with the logic, the issues I had this time were completely different. I had issues with numpy axis tricks and forcing grads to be a specific shape (why are broadcast and broadcast_to completely different methods?). I also had issues with Object Oriented Programming, as my base class Variable made every returned object a variable instead of a Tensor. I learned about Covariant Return because of this, which was an interesting design perspective. I also had a few issues learning about matrix derivatives and fixing a bug in the BCE method where my gradient update was wrong, which reminds me a bit of the issues I had with Variable exponentiation.
3. I learned about matrix derivative tricks like the trace trick. I also learned about Covariant Return and how it makes inheritance useful.
4. I would try to implement more functions like sin, cos, tanh, etc. in my autodiff library. I expect it wouldn't take too long to debug since I have numpy to rely on for the heavy lifting, but I do think it would take a bit of time. Once I start doing convnets and I make a convolution layer I might consider it.