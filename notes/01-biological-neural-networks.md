# 01 — Biological Neural Networks

> Handwritten: [pp. 45–46](../handwritten-notes/neural-network-handwritten-notes.pdf) · Exam: DUET 2018–19, Q1(a)

## Question

*Write the fundamental working principles of our information processing system / Biological Neural Network (BNN).*

## Answer

1. **A complex, adaptive processor.** The human brain is a complicated, creative information-processing system. As technology advanced, the metaphors used to describe the brain advanced with it, from hydraulics to telephone switchboards to computers.

2. **Input through the senses.** Information processing starts with the sensory organs, which **transduce** physical stimuli (touch, heat, sound waves, photons of light) into electrochemical signals carried by neurons.

3. **Bottom-up and top-down processing.** Sensory information is transformed repeatedly in both directions:
   - *bottom-up*: low-level features are assembled into structure (e.g., edges and colour changes → "a square shape"),
   - *top-down*: prior knowledge and expectations interpret those features (e.g., deciding it is a solid *black box*, not a box-shaped hole in a white wall).

4. **Attention as a filter.** Once information is partly processed, an attention mechanism decides how important a signal is and which cognitive processes receive it. It suppresses background noise (individual blades of grass) and shifts focus to salient details (hearing your own name in a noisy crowd).

5. **Storage before use.** To use new information, the brain first **encodes** it (by appearance, sound or meaning) into memory. Short-term memory is maintained by ongoing (reverberating) electrical activity; long-term memory involves physical changes in synapses, through protein synthesis, which consolidate the memory.

6. **An open research area.** How the brain processes information is still the subject of a large, ongoing body of research.

## Connection to artificial neural networks

Several of these principles have direct counterparts in the models covered later in this course and in modern deep learning:

| Biological principle | Artificial counterpart |
|---|---|
| Neurons sum weighted inputs and fire above a threshold | Perceptron ([02](02-perceptron.md)) |
| Learning by changing connection strengths (synapses) | Weight updates: delta rule, backpropagation ([05](05-backpropagation.md)) |
| Hierarchical bottom-up feature extraction | Multi-layer networks, CNNs |
| Top-down expectation checked against input | ART's top-down template and vigilance test ([09](09-adaptive-resonance-theory.md)) |
| Associative recall from partial cues | BAM ([08](08-bidirectional-associative-memory.md)) |
| Attention filter | Attention mechanisms in Transformers |
