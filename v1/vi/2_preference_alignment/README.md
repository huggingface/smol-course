# Tinh Chỉnh Theo Sự Ưu Tiên (Preference Alignment)

Trong chương này, bạn sẽ học về các kỹ thuật tinh chỉnh mô hình ngôn ngữ theo sự ưu tiên của con người. Trong khi *học có giám sát* giúp mô hình học các tác vụ, *tinh chỉnh theo sự ưu tiên* khuyến khích đầu ra phù hợp với kỳ vọng và giá trị của con người.

## Tổng Quan

Các phương pháp *tinh chỉnh theo sự ưu tiên* thường bao gồm 2 giai đoạn:

1. Bắt đầu bằng quá trình *học có giám sát* (SFT) để thích ứng mô hình với các lĩnh vực cụ thể
2. Sau đó, tinh chỉnh mô hình theo sự ưu tiên (như RLHF hoặc DPO) để cải thiện chất lượng phản hồi

Các phương pháp thay thế như ORPO kết hợp cả *tinh chỉnh theo chỉ thị* và *tinh chỉnh theo sự ưu tiên* thành 1 giai đoạn tinh chỉnh duy nhất. Ở đây, chúng ta sẽ tập trung vào các thuật toán DPO và ORPO.

Nếu bạn muốn tìm hiểu thêm về các kỹ thuật tinh chỉnh khác, bạn có thể đọc thêm tại [Argilla Blog](https://argilla.io/blog/mantisnlp-rlhf-part-8).

### 1️⃣ Tối Ưu Hóa Ưu Tiên Trực Tiếp (Direct Preference Optimization - DPO)

Phương pháp này đơn giản hóa quá trình *tinh chỉnh theo chỉ thị* bằng cách tối ưu hóa trực tiếp mô hình sử dụng dữ liệu ưu tiên (preference data). Phương pháp này loại bỏ nhu cầu về các *Mô hình thưởng phạt* (Reward model) riêng biệt và *Học tăng cường* phức tạp, giúp quá trình ổn định và hiệu quả hơn so với Học tăng cường từ phản hồi của con người (RLHF) truyền thống. Để biết thêm chi tiết, bạn có thể tham khảo tài liệu [*tối ưu hóa ưu tiên trực tiếp* (DPO)](./dpo.md).

### 2️⃣ Tối Ưu Hóa Ưu Tiên Theo Tỷ Lệ Odds (Odds Ratio Preference Optimization - ORPO)

ORPO giới thiệu một phương pháp kết hợp cả 2 giai đoạn *tinh chỉnh theo chỉ thị* và *tinh chỉnh theo sự ưu tiên* vào trong 1 giai đoạn tinh chỉnh duy nhất. Phương pháp này điều chỉnh mục tiêu tiêu chuẩn của mô hình ngôn ngữ bằng cách kết hợp *negative log-likelihood loss* với một * tỷ lệ odds* ở cấp độ *token*. Vì vậy, ORPO tạo ra 1 quá trình tinh chỉnh thống nhất với kiến trúc không cần mô hình thưởng phạt và cải thiện đáng kể hiệu quả tính toán. ORPO đã cho thấy kết quả ấn tượng trên nhiều benchmark, thể hiện hiệu suất tốt hơn trên AlpacaEval so với các phương pháp truyền thống. Để biết thêm chi tiết, bạn có thể tham khảo tài liệu [tối ưu hóa ưu tiên theo tỷ lệ odds (ORPO)](./orpo.md).

## Bài Tập

| Tiêu đề | Mô tả | Bài tập | Đường dẫn | Colab |
|-------|-------------|----------|------|-------|
| Tinh chỉnh theo DPO | Học cách tinh chỉnh mô hình bằng phương pháp tối ưu hóa ưu tiên trực tiếp | 🐢 Tinh chỉnh mô hình sử dụng bộ dữ liệu HH-RLHF <br>🐕 Sử dụng tập dữ liệu của riêng bạn<br>🦁 Thử nghiệm với các tập dữ liệu và kích thước mô hình khác nhau | [Notebook](./notebooks/dpo_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/2_preference_alignment/notebooks/dpo_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Sử dụng Colab"/></a> |
| Tinh chỉnh theo ORPO | Học cách tinh chỉnh mô hình bằng phương pháp tối ưu hóa ưu tiên theo tỷ lệ odds | 🐢 Huấn luyện mô hình sử dụng bộ dữ liệu chỉ thị (instruction) và dữ liệu ưu tiên (preference)<br>🐕 Thử nghiệm với các trọng số loss khác nhau<br>🦁 So sánh kết quả ORPO với DPO | [Notebook](./notebooks/orpo_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/2_preference_alignment/notebooks/orpo_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Sử dụng Colab"/></a> |


## Resources

- [Tài liệu thư viện TRL](https://huggingface.co/docs/trl/index) - Tài liệu cho thư viện Transformers Reinforcement Learning (TRL), triển khai nhiều kỹ thuật căn chỉnh bao gồm DPO và ORPO.
- [Bài báo nghiên cứu DPO](https://arxiv.org/abs/2305.18290) - bài nghiên cứu gốc giới thiệu *tối ưu hóa ưu tiên trực tiếp* như một giải pháp thay thế đơn giản hơn cho RLHF.
- [Bài báo nghiên cứu ORPO](https://arxiv.org/abs/2403.07691) - Giới thiệu Odds Ratio Preference Optimization, một phương pháp mới kết hợp *tinh chỉnh theo chỉ thị* và *tinh chỉnh theo sự ưu tiên* thành 1
- [Bài hướng dẫn của Argilla](https://argilla.io/blog/mantisnlp-rlhf-part-8/) - Hướng dẫn giải thích các kỹ thuật căn chỉnh khác nhau bao gồm RLHF, DPO và cách triển khai thực tế.
- [Blog về DPO](https://huggingface.co/blog/dpo-trl) - Hướng dẫn thực hành về triển khai DPO sử dụng thư viện TRL với các ví dụ code và phương pháp tốt nhất.
- [Code mẫu cho DPO trong thư viên TRL](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py) - Code mẫu về cách triển khai tinh chỉnh DPO sử dụng thư viện TRL.
- [Code mẫu cho ORPD trong thư viên TRL](https://github.com/huggingface/trl/blob/main/examples/scripts/orpo.py) - Code mẫu của tinh chỉnh ORPO sử dụng thư viện TRL với các tùy chọn cấu hình chi tiết.
- [Hugging Face Alignment Handbook](https://github.com/huggingface/alignment-handbook) - Hướng dẫn và codebase cho việc tinh chỉnh mô hình ngôn ngữ sử dụng các kỹ thuật khác nhau bao gồm SFT, DPO và RLHF.
