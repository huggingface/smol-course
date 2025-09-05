# Tinh chỉnh hiệu quả tham số (Parameter-Efficient Fine-Tuning - PEFT)

Khi các mô hình ngôn ngữ ngày càng lớn hơn, việc tinh chỉnh truyền thống trở nên ngày càng thách thức. Việc tinh chỉnh đầy đủ một mô hình với 1.7B tham số đòi hỏi bộ nhớ GPU lớn, việc lưu trữ các bản sao mô hình riêng biệt tốn kém, và có nguy cơ làm mất đi các khả năng ban đầu của mô hình. Các phương pháp tinh chỉnh hiệu quả tham số (PEFT) giải quyết những thách thức này bằng cách chỉ điều chỉnh một tập nhỏ các tham số mô hình trong khi giữ nguyên phần lớn mô hình.

Tinh chỉnh truyền thống cập nhật tất cả các tham số mô hình trong quá trình huấn luyện, điều này trở nên không khả thi với các mô hình lớn. Các phương pháp PEFT giới thiệu cách tiếp cận để điều chỉnh mô hình sử dụng ít tham số có thể huấn luyện hơn - thường ít hơn 1% kích thước mô hình gốc. Việc giảm đáng kể số lượng tham số có thể huấn luyện cho phép:

- Tinh chỉnh trên phần cứng tiêu dùng với bộ nhớ GPU hạn chế 
- Lưu trữ nhiều phiên bản điều chỉnh (adapters) cho từng tác vụ một cách hiệu quả
- Khả năng tổng quát hóa tốt hơn trong các trường hợp dữ liệu ít
- Chu kỳ huấn luyện và thử nghiệm nhanh hơn

## Các phương pháp hiện có

Trong chương này, chúng ta sẽ tìm hiểu hai phương pháp PEFT phổ biến:

### 1️⃣ Phương Pháp LoRA (Low-Rank Adaptation)

LoRA đã nổi lên như phương pháp PEFT được áp dụng rộng rãi nhất, cung cấp giải pháp hoàn hảo cho việc điều chỉnh mô hình hiệu quả mà không tốn nhiều tài nguyên tính toán. Thay vì sửa đổi toàn bộ mô hình, **LoRA đưa các ma trận có thể huấn luyện vào các lớp attention của mô hình.** Cách tiếp cận này thường giảm các tham số có thể huấn luyện khoảng 90% trong khi vẫn duy trì hiệu suất tương đương với tinh chỉnh đầy đủ. Chúng ta sẽ khám phá LoRA trong phần [LoRA (Low-Rank Adaptation)](./lora_adapters.md).

### 2️⃣ Phương Pháp Điều Chỉnh Chỉ Thị (Prompt Tuning)

Prompt tuning cung cấp cách tiếp cận **thậm chí nhẹ hơn** bằng cách **thêm các token có thể huấn luyện vào đầu vào** thay vì sửa đổi trọng số mô hình. Prompt tuning ít phổ biến hơn LoRA, nhưng có thể là kỹ thuật hữu ích để nhanh chóng điều chỉnh mô hình cho các tác vụ hoặc lĩnh vực mới. Chúng ta sẽ khám phá prompt tuning trong phần [Prompt Tuning](./prompt_tuning.md).

## Notebooks bài tập

| Tiêu đề | Mô tả | Bài tập | Link | Colab |
|---------|--------|---------|------|-------|
| Tinh chỉnh LoRA | Học cách tinh chỉnh mô hình sử dụng LoRA adapters | 🐢 Huấn luyện mô hình sử dụng LoRA<br>🐕 Thử nghiệm với các giá trị rank khác nhau<br>🦁 So sánh hiệu suất với tinh chỉnh đầy đủ | [Notebook](./notebooks/finetune_sft_peft.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/finetune_sft_peft.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Tải LoRA Adapters | Học cách tải và sử dụng LoRA adapters đã huấn luyện | 🐢 Tải adapters đã huấn luyện trước<br>🐕 Gộp adapters với mô hình cơ sở<br>🦁 Chuyển đổi giữa nhiều adapters | [Notebook](./notebooks/load_lora_adapter.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/load_lora_adapter.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## Tài liệu tham khảo
- [Tài liệu PEFT](https://huggingface.co/docs/peft)
- [Bài báo nghiên cứu LoRA](https://arxiv.org/abs/2106.09685)
- [Bài báo nghiên cứu QLoRA](https://arxiv.org/abs/2305.14314)
- [Bài báo nghiên cứu Prompt Tuning](https://arxiv.org/abs/2104.08691)
- [Hướng dẫn sử dụng PEFT của Hugging Face](https://huggingface.co/blog/peft)
- [Cách Tinh chỉnh LLM với Hugging Face](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl)
- [Thư viện TRL](https://huggingface.co/docs/trl/index)