# Tối Ưu Hóa Ưu Tiên Trực Tiếp (Direct Preference Optimization - DPO)

DPO cung cấp một cách tiếp cận đơn giản để tinh chỉnh mô hình ngôn ngữ theo ý muốn của con người. Khác với phương pháp RLHF truyền thống yêu cầu các mô hình thưởng phạt riêng biệt và học tăng cường phức tạp, DPO tối ưu hóa trực tiếp mô hình bằng dữ liệu ưu tiên (preference dataset).

## Tìm Hiểu Về DPO

DPO chuyển đổi bài toán tinh chỉnh ưu tiên thành bài toán phân loại trên dữ liệu ưu tiên của con người. Phương pháp RLHF truyền thống yêu cầu huấn luyện một mô hình thưởng phạt riêng biệt và sử dụng các thuật toán học tăng cường phức tạp như PPO để căn chỉnh đầu ra của mô hình. DPO đơn giản hóa quy trình này bằng cách định nghĩa một hàm mất mát (loss) trực tiếp tối ưu hóa chiến lược học (policy) của mô hình dựa trên các đầu ra được ưu tiên và không được ưu tiên.

Phương pháp này đã chứng minh hiệu quả cao trong thực tế, được sử dụng để huấn luyện các mô hình như Llama. Bằng cách loại bỏ nhu cầu về mô hình thưởng phạt riêng biệt và giai đoạn học tăng cường, DPO giúp việc tinh chỉnh ưu tiên trở nên dễ tiếp cận và ổn định hơn.

## DPO Hoạt Động Như Thế Nào

Quy trình DPO yêu cầu quá trình học có giám sát (SFT) để thích ứng mô hình với lĩnh vực mục tiêu. Điều này tạo nền tảng cho việc học ưu tiên bằng cách huấn luyện trên các tập dữ liệu làm theo chỉ thị tiêu chuẩn. Mô hình học cách hoàn thành tác vụ cơ bản trong khi duy trì các khả năng tổng quát.

Tiếp theo là quá trình học ưu tiên, nơi mô hình được huấn luyện trên các cặp đầu ra - một được ưu tiên và một không được ưu tiên. Các cặp ưu tiên giúp mô hình hiểu phản hồi nào phù hợp hơn với giá trị và kỳ vọng của con người.

Đổi mới cốt lõi của DPO nằm ở cách tiếp cận tối ưu hóa trực tiếp. Thay vì huấn luyện một mô hình thưởng phạt riêng biệt, DPO sử dụng hàm mất mát `binary cross-entropy` để trực tiếp cập nhật trọng số mô hình dựa trên dữ liệu ưu tiên. Quy trình đơn giản này giúp việc huấn luyện ổn định và hiệu quả hơn trong khi đạt được kết quả tương đương hoặc tốt hơn so với RLHF truyền thống.

## Bộ Dữ Liệu DPO

Bộ dữ liệu cho DPO thường được tạo bằng cách gán nhãn các cặp phản hồi là được ưu tiên hoặc không được ưu tiên. Việc này có thể được thực hiện thủ công hoặc sử dụng các kỹ thuật lọc tự động. Dưới đây là cấu trúc mẫu của tập dữ liệu preference một lượt cho DPO:

```
| Prompt | Chosen | Rejected |
|--------|--------|----------|
| ...    | ...    | ...      |
| ...    | ...    | ...      |
| ...    | ...    | ...      |
```

Cột `Prompt` chứa chỉ thị dùng để tạo ra các phản hồi. Cột `Chosen` và `Rejected` chứa các phản hồi được ưu tiên và không được ưu tiên. Có nhiều biến thể của cấu trúc này, ví dụ, bao gồm cột `System Prompt` (chỉ thị hệ thống) hoặc cột `Input` chứa tài liệu tham khảo. Giá trị của `Chosen` và `Rejected` có thể được biểu diễn dưới dạng chuỗi cho hội thoại một lượt hoặc dưới dạng danh sách hội thoại.

Bạn có thể tìm thấy bộ sưu tập các tập dữ liệu DPO trên Hugging Face [tại đây](https://huggingface.co/collections/argilla/preference-datasets-for-dpo-656f0ce6a00ad2dc33069478).

## Triển Khai Với TRL

Thư viện Transformers Reinforcement Learning (TRL) giúp việc triển khai DPO trở nên đơn giản. Các lớp `DPOConfig` và `DPOTrainer` tuân theo API của thư viện `transformers`.

Đây là ví dụ cơ bản về cách thiết lập tinh chỉnh DPO:

```python
from trl import DPOConfig, DPOTrainer

# Định nghĩa các tham số
training_args = DPOConfig(
    ...
)

# Khởi tạo trainer
trainer = DPOTrainer(
    model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    ...
)

# Huấn luyện mô hình
trainer.train()
```

Chúng ta sẽ tìm hiểu thêm chi tiết về cách sử dụng các lớp `DPOConfig` và `DPOTrainer` trong [Hướng dẫn DPO](./notebooks/dpo_finetuning_example.ipynb).

## Phương Pháp Tốt Nhất

**Chất lượng dữ liệu là yếu tố quan trọng** cho việc triển khai DPO thành công. Tập dữ liệu ưu tiên nên bao gồm các ví dụ đa dạng bao quát các khía cạnh khác nhau của hành vi mong muốn. Hướng dẫn gán nhãn rõ ràng đảm bảo việc gán nhãn nhất quán cho các phản hồi được ưu tiên và không được ưu tiên. Bạn có thể cải thiện hiệu suất mô hình bằng cách nâng cao chất lượng tập dữ liệu ưu tiên. Ví dụ, bằng cách lọc các tập dữ liệu lớn hơn để chỉ bao gồm các ví dụ chất lượng cao hoặc các ví dụ liên quan đến trường hợp sử dụng của bạn.

Trong quá trình huấn luyện, cần theo dõi kỹ sự hội tụ của hàm mất mát và đánh giá hiệu suất trên dữ liệu kiểm tra (held-out set). Tham số `beta` có thể cần điều chỉnh để cân bằng giữa việc học ưu tiên với việc duy trì các khả năng tổng quát của mô hình. Đánh giá thường xuyên trên các chỉ thị đa dạng giúp đảm bảo mô hình đang học các ưu tiên mong muốn mà không bị tình trạng quá khớp (overfitting).

So sánh đầu ra của mô hình với mô hình tham chiếu để xác minh sự cải thiện trong tinh chỉnh ưu tiên. Kiểm tra trên nhiều chỉ thị khác nhau, bao gồm cả các trường hợp ngoại lệ (edge cases), giúp đảm bảo việc học ưu tiên mạnh mẽ hơn trong các tình huống khác nhau.

## Các Bước Tiếp Theo

⏩ Để có thể thực hành thực tế với DPO, hãy thử [Hướng dẫn DPO](./notebooks/dpo_finetuning_example.ipynb). Hướng dẫn thực hành này sẽ chỉ dẫn bạn cách triển khai tinh chỉnh ưu tiên với mô hình của riêng bạn, từ chuẩn bị dữ liệu đến huấn luyện và đánh giá.

⏭️ Sau khi hoàn thành hướng dẫn, bạn có thể khám phá về [ORPO](./orpo.md) để tìm hiểu về một kỹ thuật tinh chỉnh ưu tiên khác.