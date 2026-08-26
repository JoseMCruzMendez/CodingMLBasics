# What I learned coding RNNs

## Before I Start (In post this time):

My ML studies had never really gone past classification, and even then I had never classified series data before so I had never used sequence-based models like RNNs before. It was quite interesting to learn about them, and I took it as a challenge to look up some history on them to try to piece together how Transformers came about. It was a bit hard since the recurrent connections were easy to understand conceptually but harder to implement than I anticipated, but this chapter was overall pretty fun.

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

1. I attempted to use RNNs to predict simple exponential decay and the alphabet. I also tried to use auto-regression to predict patterns. I had ok accuracy in the exponential case and great accuracy in the alphabet case. Auto-regression wise, the exponential case was not that good, although I had some hyperparameter setups that could predict some decay patterns rather well. The alphabet case was perfect up to 1000 characters generated, much more than the training data's 26 characters.
2. It was quite challenging to implement RNNs at first, as I was confused on how exactly the self-connection should be implemented even if I understood it at a conceptual level. I was also surprised at how hard it was to get the model to work, and all the architectural difficulties I ran into while trying to set up the RNN. For example, if I didn't shuffle the alphabet sequence order, the model would overfit to "bcd...za" and exclusively output that, getting lost after "za...". I also tried implementing ExpRNN to force the self-relation matrix to be orthogonal to prevent gradient explosions/vanishings, to no success. 
3. I learned about RNNs, and how they are quite adept at learning sequential data. I also learned how part of their utility comes from auto-regression and being able to predict sequences. Finally, I learned how RNNs lead to specific architectural issues like gradient explosions or vanishing gradients, and how this motivated more sophisticated architectures like LSTMs or GRUs.
4. If I had infinite time, I would try to finish implementing ExpRNN and trying to learn more complex data. I would also like to know how to get the auto-regression test to work a little better in the exponential decay task, as I believe by modifying the training loop I would be able to get better results.

IN POST: I realized the alphabet example was probably too simple for the RNN, as a neural network should be able to map a->b->c... etc. So I tried implementing the problem that ExpRNN was tested on, which is a simple memory problem. I got results which where quite good, although I did have to reduce the message length quite a bit for it to work. I started with 20 characters, but quickly saw that it was too long for my small RNN, so I pushed it down to about 12 and got good results.