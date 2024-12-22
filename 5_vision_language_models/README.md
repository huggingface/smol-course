# Vision Language Models

## 1. VLM Usage

Vision Language Models (VLMs) process image inputs alongside text to enable tasks like image captioning, visual question answering, and multimodal reasoning.  

A typical VLM architecture consists of an image encoder to extract visual features, a projection layer to align visual and textual representations, and a language model to process or generate text. This allows the model to establish connections between visual elements and language concepts.

VLMs can be used in different configurations depending on the use case. Base models handle general vision-language tasks, while chat-optimized variants support conversational interactions. Some models include additional components for grounding predictions in visual evidence or specializing in specific tasks like object detection.

For more on the technicality and usage of VLMs, refer to the [VLM Usage](./vlm_usage.md) page.  

## 2. VLM Fine-Tuning  

Fine-tuning a VLM involves adapting a pre-trained model to perform specific tasks or to operate effectively on a particular dataset. The process can follow methodologies such as supervised fine-tuning, preference optimization, or a hybrid approach that combines both, as introduced in Modules 1 and 2.  

While the core tools and techniques remain similar to those used for LLMs, fine-tuning VLMs requires additional focus on data representation and preparation for images. This ensures the model effectively integrates and processes both visual and textual data for optimal performance. Given that the demo model, SmolVLM, is significantly larger than the language model used in the previous module, it's essential to explore methods for efficient fine-tuning. Techniques like quantization and PEFT can help make the process more accessible and cost-effective, allowing more users to experiment with the model. 

For detailed guidance on fine-tuning VLMs, visit the [VLM Fine-Tuning](./vlm_finetuning.md) page.  


## Exercise Notebooks  


| Title | Description | Exercise | Link | Colab |
|-------|-------------|----------|------|-------|
| VLM Usage | Learn how to load and use a pre-trained VLM for various tasks | 🐢 Process an image<br>🐕 Process multiple images with batch handling <br>🦁 Process a full video| [Notebook](./notebooks/vlm_usage_sample.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/5_vision_language_models/notebooks/vlm_usage_sample.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| VLM Fine-Tuning | Learn how to fine-tune a pre-trained VLM for task-specific datasets | 🐢 Use a basic dataset for fine-tuning<br>🐕 Try a new dataset<br>🦁 Experiment with alternative fine-tuning methods | [Notebook](./notebooks/vlm_sft_sample.ipynb)| <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/5_vision_language_models/notebooks/vlm_sft_sample.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | 


## References  
- [Hugging Face Learn: Supervised Fine-Tuning VLMs](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl)
- [Hugging Face Learn: Supervised Fine-Tuning SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_smol_vlm_sft_trl)  
- [Hugging Face Learn: Preference Optimization Fine-Tuning SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_vlm_dpo_smolvlm_instruct)  
- [Hugging Face Blog: Preference Optimization for VLMs](https://huggingface.co/blog/dpo_vlm)
- [Hugging Face Blog: Vision Language Models](https://huggingface.co/blog/vlms)
- [Hugging Face Blog: SmolVLM](https://huggingface.co/blog/smolvlm)  
- [Hugging Face Model: SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
- [CLIP: Learning Transferable Visual Models from Natural Language Supervision](https://arxiv.org/abs/2103.00020)  
- [Align Before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651)  
