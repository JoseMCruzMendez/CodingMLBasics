# What I learned coding LSTM/GRU

## Before I Start:

I have heard quite a bit related to the shortcomings of RNNs during my work on the previous chapter, things like vanishing/exploding gradients or how performance is "bottlenecked" by the last hidden state. I ran into the attention mechanism in my studies, but I decided to skip it's implementation and go straight to transformers after this. I am looking forward to deciphering all of these cryptic diagrams explaining how to implement LSTM/GRUs. I originally intended to make both of them their own separate units, but they are so inter-related that I thought I could tackle them in a single one.

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

1. I implemented LSTMs and GRUs, running similar experiments to the ones I did with RNNs (exponential decay, memory tests). I made tests harder this time: other than decay I was also testing exponential growth, and now messages were of different lengths every time. LSTMs and GRUs had a much better time training than vanilla RNNs, although I still had some issues with "teacher forcing" biasing the network towards poor autoregressive behavior. Message wise, nets learned very well and adapted quickly to message length, while taking more time to accurately represent message content.
2. What surprised me was just how many variations of LSTM/GRUs there are out there. Maybe I was consulting the wrong sources, but there were many different architectures and schemes to join input and hidden state. I also had a little bit of a hard time reading diagrams because of this, since some people concatenated where others added and so on. This has been a first, as in all other modules the architecture I was studying was very well established. I was surprised there was so much variation, since LSTMs/GRUs are responsible for a lot of results related to sequential processing, and I expected them to be more standardized.
3. I learned about LSTMs/GRUs, and how their approaches to memory help with gradient explosions/vanishings, and help keep relevant content in memory for longer.
4. If I had infinite time, I would try "professor forcing", and adversarial approach to generating with recurrent architectures, that promises to help with the poor autoregression results. I would also try to implement attention and seeing if that helps performance in the message scenario.