# Tinh chỉnh theo chỉ thị (Instruction Tuning)

Trong chương này chúng ta sẽ học về quá trình tinh chỉnh mô hình ngôn ngữ theo chỉ thị. Tinh chỉnh theo chỉ thị là quá trình điều chỉnh *pre-trained models* cho các tác vụ cụ thể bằng cách tiếp tục huấn luyện chúng trên các tập dữ liệu đặc thù cho tác vụ. Quá trình này giúp các mô hình cải thiện hiệu suất trên những tác vụ đó.

Chúng ta sẽ cùng khám phá hai chủ đề chính: 1) Định dạng Chat (Chat Templates) và 2) Tinh chỉnh có giám sát (Supervised Fine-Tuning).

## 1️⃣ Định dạng Chat (Chat Templates)

Định dạng Chat là cấu trúc giữa các tương tác giữa người dùng và mô hình ngôn ngữ, đảm bảo các phản hồi nhất quán và phù hợp với từng ngữ cảnh. Chúng bao gồm các thành phần như `system prompts` và các `message` theo vai trò (người dùng - `user` hoặc trợ lý - `assistant`). Để biết thêm thông tin chi tiết, hãy tham khảo phần [Chat Templates](./chat_templates.md).

## 2️⃣ Huấn luyện có giám sát (Supervised Fine-Tuning)

Huấn luyện có giám sát (SFT) là một quá trình cốt lõi để điều chỉnh các mô hình ngôn ngữ đã *pre-trained* cho các tác vụ cụ thể. Quá trình này bao gồm việc huấn luyện mô hình trên tập dữ liệu có gán nhãn theo tác vụ cụ thể. Để đọc hướng dẫn chi tiết về SFT, bao gồm các bước quan trọng và các phương pháp thực hành tốt nhất, hãy xem tại trang [Supervised Fine-Tuning](./supervised_fine_tuning.md).

## Tài liệu tham khảo

- [Tài liệu Transformers về định dạng chat](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Script cho huấn luyện có giám sát bằng thư viện TRL](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py)
- [`SFTTrainer` trong thư viện TRL](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Bài báo Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)
- [Huấn luyện có giám sát bằng thư viện TRL](https://huggingface.co/docs/trl/main/en/tutorials/supervised_finetuning)
- [Cách fine-tune Google Gemma với ChatML và Hugging Face TRL](https://www.philschmid.de/fine-tune-google-gemma)
- [Huấn luyện LLM để tạo danh mục sản phẩm tiếng Ba Tư ở định dạng JSON](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format)
