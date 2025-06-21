# What I learned coding Convnets

## Before I Start:

I have never used a convnet before despite dabbling a bit in ML architectures. I simply have never worked with natural 2D data like images, so I've never found a use for them. I admit before reading about what they do I found them a little mystifying and thus intimidating to implement. Now that I know about what they are supposed to do, I think I will be able to implement them. I hope I don't run into broadcasting problems again, especially since I will probably end up using stride-tricks.

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

1. I managed to implement a convnet that mimics LeNet-5, and obtain good accuracy on MNIST with significantly less epochs and parameters than the fully connected Neural Network I was using for the NN tests. I managed to push accuracy up to 87%, but that was only because the model had ~95% accuracy on everything except 6, where it had 0%.
2. Once again broadcasting was a pain. I first tried to code a fully broadcasted convolution, but quickly found it to be unwieldy. I ended up resorting to for loops, but that makes my convnet incredibly slow. As in ~25x slower than my neural net implementation. I really hope broadcasting is not an issue for the following modules because I am *this* close to ditching my Autograd and going to torch. I will stick with it at least until the end of this module, I do want to see how far I can take it. The for loop issue probably means my attention implementation will be snail slow though. Because training took so long I tried experimenting with a few hyperparameter setups, but always found that my net either misclassified one or two categories at 100% rate which pushed accuracy down or classified everything as a single category.
3. I re-discovered the power of convolutions in finding patterns in images, being translation-invariant, and overall learning "visual" patterns much faster than traditional neural networks.
4. Try to implement the fully vectorized version of convolution to speed up tests. I think I will end up testing convergence speed in torch after this unit. If time permits, I would also like to implement dropout and batch normalization to speed up training, although the current pace at which my network goes through data is a roadblock to quick testing.