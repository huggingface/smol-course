# Huấn luyện theo chỉ thị (Instruction Tuning)

Trong bài này sẽ chúng ta sẽ học về quá trình huấn luyện mô hình ngôn ngữ theo chỉ thị. Huấn luyện theo chỉ thị là quá trình điều chỉnh các mô hình đã được *pre-trained* cho các tác vụ cụ thể bằng cách huấn luyện thêm chúng trên các tập dữ liệu theo tác vụ. Quá trình này giúp các mô hình cải thiện hiệu suất trên những tác vụ mục tiêu.

Chúng ta sẽ cùng khám phá hai chủ đề chính: 1) Chat Templates và 2) Huấn luyện có giám sát (Supervised Fine-Tuning).

## 1️⃣ Chat Templates

Chat templates là cấu trúc giữa các tương tác giữa người dùng và mô hình ngôn ngữ, đảm bảo các phản hồi nhất quán và phù hợp với từng ngữ cảnh. Chúng bao gồm các thành phần như `system prompts` và các `message` theo vai trò (người dùng - `user` hoặc trợ lý - `assistant`). Để biết thêm thông tin chi tiết, hãy tham khảo phần [Chat Templates](./chat_templates.md).

## 2️⃣ Huấn luyện có giám sát (Supervised Fine-Tuning)

Huấn luyện có giám sát (SFT) là một quá trình cốt lõi để điều chỉnh các mô hình ngôn ngữ đã *pre-trained* cho các tác vụ cụ thể. Quá trình này bao gồm việc huấn luyện mô hình trên tập dữ liệu có gán nhãn theo tác vụ cụ thể. Để đọc hướng dẫn chi tiết về SFT, bao gồm các bước quan trọng và các phương pháp thực hành tốt nhất, hãy xem tại trang [Supervised Fine-Tuning](./supervised_fine_tuning.md).

## Bài tập

| Tiêu đề | Mô tả | Bài tập | Đường dẫn | Google Colab |
|-------|-------------|----------|------|-------|
| Định dạng Chat | Học cách sử dụng *định dạng chat* với SmolLM2 và xử lý dữ liệu thành định dạng *chatml* | 🐢 Chuyển đổi tập dữ liệu `HuggingFaceTB/smoltalk` sang định dạng *chatml* <br> 🐕 Chuyển đổi tập dữ liệu `openai/gsm8k` sang định dạng *chatml* | [Notebook](./notebooks/chat_templates_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/1_instruction_tuning/notebooks/chat_templates_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Mở trong Colab"/></a> |
| Huấn luyện có giám sát | Học cách huấn luyện SmolLM2 sử dụng SFTTrainer | 🐢 Sử dụng tập dữ liệu `HuggingFaceTB/smoltalk` <br>🐕 Thử nghiệm với tập dữ liệu `bigcode/the-stack-smol` <br>🦁 Chọn một tập dữ liệu cho trường hợp sử dụng thực tế | [Notebook](./notebooks/sft_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/1_instruction_tuning/notebooks/sft_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Mở trong Colab"/></a> |

## Tài liệu tham khảo

- [Tài liệu Transformers về chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Script cho huấn luyện có giám sát bằng thư viện TRL](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py)
- [`SFTTrainer` trong thư viện TRL](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Bài báo Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)
- [Huấn luyện có giám sát bằng thư viện TRL](https://huggingface.co/docs/trl/main/en/tutorials/supervised_finetuning)
- [Cách fine-tune Google Gemma với ChatML và Hugging Face TRL](https://www.philschmid.de/fine-tune-google-gemma)
- [Huấn luyện LLM để tạo danh mục sản phẩm tiếng Ba Tư ở định dạng JSON](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format)
