# Mô hình Ngôn ngữ Thị giác

## 1. Sử dụng Mô hình Ngôn ngữ Thị giác (Vision Language Models)

Mô hình Ngôn ngữ Thị giác (VLMs) xử lý đầu vào hình ảnh cùng với văn bản để thực hiện các tác vụ như chú thích của ảnh, trả lời câu hỏi bằng hình ảnh và suy luận đa phương thức (multimodal).

Một kiến trúc VLM điển hình bao gồm:
1. Bộ mã hóa hình ảnh (*image encoder*) để trích xuất các đặc trưng thị giác
2. Lớp chiếu (*projection layer*) để căn chỉnh các biểu diễn thị giác với văn bản
3. Mô hình ngôn ngữ để xử lý hoặc tạo văn bản. Điều này cho phép mô hình thiết lập các kết nối giữa các yếu tố về thị giác và các khái niệm trong ngôn ngữ.

Tùy thuộc vào từng trường hợp mà có thể sử dụng các VLMs được huấn luyện theo các tác vụ khác nhau. Các mô hình cơ sở (base models) xử lý các tác vụ thị giác-ngôn ngữ tổng quát, trong khi các biến thể tối ưu hóa cho trò chuyện (chat-optimized variants) hỗ trợ các tương tác hội thoại. Một số mô hình bao gồm các thành phần bổ sung để làm rõ dự đoán dựa trên các bằng chứng thị giác (*visual evidence*) hoặc chuyên về các tác vụ cụ thể như phát hiện đối tượng (*object detection*).

Để biết thêm chi tiết về kỹ thuật và cách sử dụng VLMs, hãy tham khảo trang [Sử dụng VLM](./vlm_usage.md).

## 2. Tinh chỉnh Mô hình Ngôn ngữ Thị giác (VLM)

Tinh chỉnh VLM là việc điều chỉnh một mô hình đã được huấn luyện trước (*pre-trained*) để thực hiện các tác vụ cụ thể hoặc để hoạt động hiệu quả trên một tập dữ liệu cụ thể. Quá trình này có thể tuân theo các phương pháp như tinh chỉnh có giám sát (*supervised fine-tuning*), tối ưu hóa tùy chọn (*preference optimization*) hoặc phương pháp kết hợp (*hybrid approach*) cả hai, như đã giới thiệu trong Chương 1 và Chương 2.

Mặc dù các công cụ và kỹ thuật cốt lõi vẫn tương tự như các công cụ và kỹ thuật được sử dụng cho các mô hình ngôn ngữ (LLMs), việc tinh chỉnh VLMs đòi hỏi phải tập trung nhiều hơn vào việc biểu diễn và chuẩn bị dữ liệu cho hình ảnh. Điều này đảm bảo mô hình tích hợp và xử lý hiệu quả cả dữ liệu thị giác và văn bản để đạt hiệu suất tối ưu. Vì mô hình demo, SmolVLM, lớn hơn đáng kể so với mô hình ngôn ngữ được sử dụng trong bài trước, điều cần thiết là phải khám phá các phương pháp tinh chỉnh hiệu quả. Các kỹ thuật như lượng tử hóa (*quantization*) và Tinh chỉnh hiệu quả tham số - PEFT (*Parameter-Efficient Fine-Tuning*) có thể giúp làm cho quá trình này dễ tiếp cận hơn và tiết kiệm chi phí hơn, cho phép nhiều người dùng thử nghiệm với mô hình hơn.

Để được hướng dẫn chi tiết về tinh chỉnh VLMs, hãy truy cập trang [Tinh chỉnh VLM](./vlm_finetuning.md).

## Bài tập thực hành

| Tiêu đề | Mô tả | Bài tập | Link | Colab |
|-------|-------------|----------|------|-------|
| Sử dụng VLM | Tìm hiểu cách tải và sử dụng VLM đã được huấn luyện trước cho các tác vụ khác nhau | 🐢 Xử lý một hình ảnh<br>🐕 Xử lý nhiều hình ảnh với xử lý hàng loạt <br>🦁 Xử lý toàn bộ video| [Notebook](./notebooks/vlm_usage_sample.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/5_vision_language_models/notebooks/vlm_usage_sample.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Tinh chỉnh VLM | Tìm hiểu cách tinh chỉnh VLM đã được huấn luyện trước cho các tập dữ liệu theo từng nhiệm vụ | 🐢 Sử dụng tập dữ liệu cơ bản để tinh chỉnh<br>🐕 Thử tập dữ liệu mới<br>🦁 Thử nghiệm với các phương pháp tinh chỉnh thay thế | [Notebook](./notebooks/vlm_sft_sample.ipynb)| <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/5_vision_language_models/notebooks/vlm_sft_sample.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## Tài liệu tham khảo

- [Hugging Face Learn: Tinh chỉnh có giám sát VLMs](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl)
- [Hugging Face Learn: Tinh chỉnh có giám sát SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_smol_vlm_sft_trl)
- [Hugging Face Learn: Tinh chỉnh tối ưu hóa tùy chọn SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_vlm_dpo_smolvlm_instruct)
- [Hugging Face Blog: Tối ưu hóa tùy chọn cho VLMs](https://huggingface.co/blog/dpo_vlm)
- [Hugging Face Blog: Mô hình Ngôn ngữ Thị giác](https://huggingface.co/blog/vlms)
- [Hugging Face Blog: SmolVLM](https://huggingface.co/blog/smolvlm)
- [Hugging Face Model: SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
- [CLIP: Learning Transferable Visual Models from Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Align Before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651)
