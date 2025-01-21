# Quantization Fundamentals

## What is Quantization?
Quantization is a technique used to reduce memory and computational costs by representing model weights and activations with lower-precision data types, such as 8-bit integers (int8). By doing so, it allows larger models to fit into memory and speeds up inference, making the model more efficient without significantly sacrificing performance.

* motivation - already wrote
* Floating Point Representation dtypes - float32, float16, bfloat, int8, int 4
* absmax &  zero-point quantization
* handling outliers with float16

## Quantization Techniques
* Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT) - focus on PTQ

## Quantization for Inference
Need to look for more resources

## Exercise Notebooks
I'm unsure about what exactly we should include here. Below are a few options, along with my humble thoughts:
* Type casting (float32 to int8): This seems too low-level.
* Reproducing a GPT-2 example from the Maxime blog post: I'm uncertain about the contribution this would make.
* Taking a large model from the Hugging Face Hub and converting it to a quantized model: This might fit better in the section where we discuss GTPQ or other quantization methods.

## Open Questions
Where should we talk about "quantization method" like gptq?

## References
https://huggingface.co/docs/transformers/main_classes/quantization
https://huggingface.co/docs/transformers/v4.48.0/quantization/overview
https://huggingface.co/docs/optimum/en/concept_guides/quantization
https://huggingface.co/blog/introduction-to-ggml
https://huggingface.co/docs/hub/gguf
https://huggingface.co/docs/transformers/gguf
