# Tạo tập dữ liệu giả lập (Synthetic Datasets)

Dữ liệu giả lập (synthetic data) là dữ liệu được tạo ra nhân tạo mô phỏng việc sử dụng trong thế giới thực. Nó cho phép khắc phục các hạn chế về dữ liệu bằng cách mở rộng hoặc nâng cao các tập dữ liệu. Mặc dù dữ liệu giả lập đã được sử dụng cho một số trường hợp, các mô hình ngôn ngữ lớn đã làm cho các tập dữ liệu giả lập trở nên phổ biến hơn cho việc huấn luyện trước, huấn luyện sau và đánh giá các mô hình ngôn ngữ.

Chúng ta sẽ sử dụng [`distilabel`](https://distilabel.argilla.io/latest/), một thư viện (framework) tạo dữ liệu giả lập và phản hồi AI cho các kỹ sư, những người cần các quy trình (pipeline) nhanh, đáng tin cậy và có thể mở rộng dựa trên các bài báo nghiên cứu đã được xác minh. Để tìm hiểu sâu hơn về package và các phương pháp hay nhất, hãy xem [tài liệu](https://distilabel.argilla.io/latest/).

## Tổng quan về Mô-đun

Dữ liệu giả lập cho các mô hình ngôn ngữ có thể được phân loại thành ba loại: hướng dẫn (instructions), sở thích (preferences) và phê bình (critiques). Chúng ta sẽ tập trung vào hai loại đầu tiên, tập trung vào việc tạo ra các tập dữ liệu để tinh chỉnh hướng dẫn (instruction tuning) và điều chỉnh sở thích (preference alignment). Trong cả hai loại, chúng ta sẽ đề cập đến các khía cạnh của loại thứ ba, tập trung vào việc cải thiện dữ liệu hiện có bằng các phê bình và viết lại của mô hình.

![Phân loại dữ liệu giả lập](./images/taxonomy-synthetic-data.png)

## Nội dung

### 1. [Tập dữ liệu hướng dẫn](./instruction_datasets.md)

Tìm hiểu cách tạo tập dữ liệu hướng dẫn để tinh chỉnh hướng dẫn. Chúng ta sẽ khám phá việc tạo các tập dữ liệu tinh chỉnh hướng dẫn thông qua các lời nhắc (prompting) cơ bản và sử dụng các kỹ thuật nhắc nhở tinh tế hơn từ các bài báo. Các tập dữ liệu tinh chỉnh hướng dẫn với dữ liệu mẫu (seed data) để học trong ngữ cảnh (in-context learning) có thể được tạo ra thông qua các phương pháp như `SelfInstruct` và `Magpie`. Ngoài ra, chúng ta sẽ khám phá sự tiến hóa hướng dẫn thông qua `EvolInstruct`. [Bắt đầu học](./instruction_datasets.md).

### 2. [Tập dữ liệu ưu tiên](./preference_datasets.md)

Tìm hiểu cách tạo tập dữ liệu sở thích để điều chỉnh sở thích. Chúng ta sẽ xây dựng dựa trên các phương pháp và kỹ thuật được giới thiệu trong phần 1, bằng cách tạo thêm các phản hồi. Tiếp theo, chúng ta sẽ học cách cải thiện các phản hồi đó bằng lời nhắc `EvolQuality`. Cuối cùng, chúng ta sẽ khám phá cách đánh giá các phản hồi bằng lời nhắc `UltraFeedback`, lời nhắc này sẽ tạo ra điểm số và phê bình, cho phép chúng ta tạo các cặp sở thích. [Bắt đầu học](./preference_datasets.md).

### Notebook bài tập

| Tiêu đề | Mô tả | Bài tập | Liên kết | Colab |
|-------|-------------|----------|------|-------|
| Tập dữ liệu hướng dẫn | Tạo tập dữ liệu để tinh chỉnh hướng dẫn | 🐢 Tạo tập dữ liệu tinh chỉnh hướng dẫn <br> 🐕 Tạo tập dữ liệu tinh chỉnh hướng dẫn với dữ liệu hạt giống <br> 🦁 Tạo tập dữ liệu tinh chỉnh hướng dẫn với dữ liệu hạt giống và với sự tiến hóa hướng dẫn | [Liên kết](./notebooks/instruction_sft_dataset.ipynb) | [Colab](https://githubtocolab.com/huggingface/smol-course/tree/main/6_synthetic_datasets/notebooks/instruction_sft_dataset.ipynb) |
| Tập dữ liệu ưu tiên | Tạo tập dữ liệu để điều chỉnh sở thích | 🐢 Tạo tập dữ liệu điều chỉnh sở thích <br> 🐕 Tạo tập dữ liệu điều chỉnh sở thích với sự tiến hóa phản hồi <br> 🦁 Tạo tập dữ liệu điều chỉnh sở thích với sự tiến hóa phản hồi và phê bình | [Liên kết](./notebooks/preference_alignment_dataset.ipynb) | [Colab](https://githubtocolab.com/huggingface/smol-course/tree/main/6_synthetic_datasets/notebooks/preference_alignment_dataset.ipynb) |

## Tài liệu tham khảo

- [Tài liệu Distilabel](https://distilabel.argilla.io/latest/)
- [Trình tạo dữ liệu tổng hợp là ứng dụng UI](https://huggingface.co/blog/synthetic-data-generator)
- [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
- [Self-instruct](https://arxiv.org/abs/2212.10560)
- [Evol-Instruct](https://arxiv.org/abs/2304.12244)
- [Magpie](https://arxiv.org/abs/2406.08464)
- [UltraFeedback](https://arxiv.org/abs/2310.01377)
- [Deita](https://arxiv.org/abs/2312.15685)