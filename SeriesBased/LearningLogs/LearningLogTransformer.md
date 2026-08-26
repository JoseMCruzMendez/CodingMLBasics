# What I learned coding Transformers

## Before I Start:

Before I fully fleshed out the "syllabus" for this project, the Transformer architecture was meant as a capstone. Although I extended the syllabus to include basic graph methods and XGBoost to some extent, I still see the transformer as a sort of "capstone", and I believe it is a relevant architecture for a project I am participating in soon, so I am very excited to implement and understand the architecture. This feels like a personal achievement, as I've had "understanding transformers" as something I've wanted to do for a few years now, as generic as the release of ChatGPT is of a motivation for the task. Being able to look back on everything that has happened, and knowing it is no longer an idle curiosity but something within my grasp now is quite exciting.

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

1. I implemented a transformer architecture and ran it on the same tasks I tested RNNs/LSTMs/GRUs on. To be specific, I ran it on a message memory test to check if it had memory-like capabilities, and I ran it on autoregression for time series. The transformer could have probably been the quickest architecture to get 100% on the memory task, had the lr annealing been automatic instead of manual. It was also the only architecture I tested that did not experience mode collapse with basic teacher forcing methods. Interestingly, it still had _some_ level of collapse as every single exponential curve regressed towards the mean for a few steps before proceeding in the expected manner. 
2. I should have seen this coming after my experience with RNNs, but conceptually understanding how attention works did not lead to an easier time implementing the code. This was definitely the hardest architecture to implement so far, and by a long shot. Thankfully, I now have more experience with tensor reshaping and other operations, so once the path forward was clear it was just a matter of getting things to work. There were a lot of small bugs I had to iron out in my implementation, mostly due to the amount of moving parts. Getting it to work felt really rewarding. Although there were a lot of challenging parts I don't think there was any specific part that gave me the most trouble other than trying to understand how the encoder/decoder towers interacted based on the figures from "Attention is all you need."
3. I learned how we can transition from recurrent models to feed-forward models through the use of attention, and how this idea made transformers come to the forefront of current machine learning advances in seq2seq environments. I also learned that I can use only encoder blocks if I don't need any generation, and that attention is a universal approximator even without the FNN between layers.
4. If I had infinite time, I would try to experiment with different attention patterns, try to implement KV caching so that generation did not take forever, and try other transformer-inspired architectures like informers or universal transformers.