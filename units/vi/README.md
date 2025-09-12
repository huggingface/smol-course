![smolcourse image](../banner.png)

# Khoá học Mô hình ngôn ngữ cơ bản

Đây là một khoá học thực hành về việc huấn luyện các mô hình ngôn ngữ (LM) cho các trường hợp sử dụng cụ thể. Khoá học này là cách thuận tiện để bắt đầu với việc điều chỉnh các mô hình ngôn ngữ, bởi vì mọi thứ đều có thể chạy được trên hầu hết các máy tính cá nhân. Tại đây, chúng ta không cần quá nhiều tài nguyên GPUs hay các dịch vụ trả phí để học tập. Khoá học được xây dựng dựa trên series mô hình [SmolLM2](https://github.com/huggingface/smollm/tree/main), nhưng bạn có thể áp dụng các kỹ năng học được ở đây cho các mô hình lớn hơn hoặc các mô hình ngôn ngữ nhỏ khác.

*Lưu ý: Vì khóa học được dịch từ bản gốc tiếng Anh, chúng tôi sẽ giữ lại một số thuật ngữ gốc nhằm tránh gây hiểu lầm.*

<a href="http://hf.co/join/discord">s
<img src="https://img.shields.io/badge/Discord-7289DA?&logo=discord&logoColor=white"/>
</a>

<div style="background: linear-gradient(to right, #e0f7fa, #e1bee7, orange); padding: 20px; border-radius: 5px; margin-bottom: 20px; color: purple;">
    <h2>Tham gia học ngay!</h2>
    <p>Khoá học này mở và được đánh giá bởi cộng đồng. Để tham gia khoá học <strong>hãy tạo Pull Request (PR)</strong> và gửi bài làm của bạn để được review. Các bước thực hiện:</p>
    <ol>
        <li>Fork repo <a href="https://github.com/huggingface/smol-course/fork">tại đây</a></li>
        <li>Đọc tài liệu, thực hiện thay đổi, làm bài tập, thêm ví dụ của riêng bạn</li>
        <li>Tạo PR trên nhánh december_2024</li>
        <li>Nhận review và merge</li>
    </ol>
    <p>Điều này sẽ giúp bạn học tập và xây dựng một khoá học có cộng đồng tham gia và luôn được cải thiện.</p>
</div>

Chúng ta có thể thảo luận về quá trình học tập và phát triển trong [thread này](https://github.com/huggingface/smol-course/discussions/2#discussion-7602932).

## Nội dung khoá học

Khoá học này cung cấp phương pháp thực hành để làm việc với các mô hình ngôn ngữ nhỏ, từ huấn luyện ban đầu đến triển khai lên sản phẩm.

| Bài | Mô tả | Trạng thái | Ngày phát hành |
|--------|-------------|---------|--------------|
| [Tinh chỉnh theo chỉ thị (Instruction Tuning)](./1_instruction_tuning) | Học về huấn luyện có giám sát (SFT), định dạng chat, và thực hiện các chỉ thị cơ bản | ✅ Sẵn sàng | 3/12/2024 |
| [Tinh chỉnh theo sự ưu tiên (Preference Alignment)](./2_preference_alignment) | Học các kỹ thuật DPO và ORPO để huấn luyện mô hình theo sự uy tiên của người dùng | ✅ Sẵn sàng  | 6/12/2024 |
| [Tinh chỉnh hiệu quả tham số (Parameter-efficient Fine-tuning)](./3_parameter_efficient_finetuning) | Học về LoRA, prompt tuning và các phương pháp huấn luyện hiệu quả | ✅ Sẵn sàng | 9/12/2024 |
| [Đánh giá mô hình (Evaluation)](./4_evaluation) | Sử dụng benchmark tự động và tạo đánh giá theo lĩnh vực cụ thể | [🚧 Đang thực hiện](https://github.com/huggingface/smol-course/issues/42) | 13/12/2024 |
| [Mô hình đa phương thức (Vision-language Models)](./5_vision_language_models) | Điều chỉnh các mô hình đa phương thức (Multimodal models) cho các tác vụ thị giác-ngôn ngữ | [🚧 Đang thực hiện](https://github.com/huggingface/smol-course/issues/49) | 16/12/2024 |
| [Dữ liệu nhân tạo (Synthetic Datasets)](./6_synthetic_datasets) | Tạo và đánh giá tập dữ liệu tổng hợp cho huấn luyện | 📝 Đã lên kế hoạch | 20/12/2024 |
| [Triển khai mô hình (Inference)](./7_inference) | Triển khai mô hình một cách hiệu quả | 📝 Đã lên kế hoạch | 23/12/2024 |

## Tại sao lại chọn Mô hình Ngôn ngữ Nhỏ?

Mặc dù các mô hình ngôn ngữ lớn đã cho thấy khả năng ấn tượng, chúng thường yêu cầu tài nguyên tính toán đáng kể và có thể quá mức cần thiết cho các ứng dụng tập trung. Các mô hình ngôn ngữ nhỏ mang lại nhiều lợi thế cho các ứng dụng theo lĩnh vực cụ thể:

- **Hiệu quả:** Yêu cầu ít tài nguyên tính toán hơn đáng kể để huấn luyện và triển khai
- **Tùy chỉnh:** Dễ dàng tinh chỉnh phù hợp cho các lĩnh vực cụ thể
- **Kiểm soát:** Dễ dàng hiểu và kiểm soát hành vi mô hình hơn
- **Chi phí:** Chi phí huấn luyện và triển khai thấp hơn đáng kể
- **Bảo mật:** Có thể triển khai trên mạng cục bộ mà không cần gửi dữ liệu tới API bên ngoài
- **Công nghệ xanh:** Sử dụng tài nguyên hiệu quả (carbon footprint giảm)
- **Nghiên cứu học thuật dễ dàng hơn:** Khởi đầu dễ dàng cho nghiên cứu học thuật trên mô hình ngôn ngữ với ít ràng buộc về tài nghiên tính toán

## Yêu cầu tiên quyết

Trước khi bắt đầu, hãy đảm bảo bạn có:
- Hiểu biết cơ bản về machine learning (ML) và xử lý ngôn ngữ tự nhiên (NLP)
- Quen thuộc với Python, PyTorch và thư viện `transformers`
- Hiểu về mô hình ngôn ngữ và tập dữ liệu có nhãn

## Cài đặt

Chúng tôi duy trì khoá học như một package để bạn có thể cài đặt dễ dàng thông qua package manager. Chúng tôi gợi ý sử dụng [uv](https://github.com/astral-sh/uv), nhưng bạn có thể sử dụng các lựa chọn thay thế như `pip` hoặc `pdm`.

### Sử dụng `uv`

Với `uv` đã được cài đặt, bạn có thể cài đặt khoá học như sau:

```bash
uv venv --python 3.11.0
uv sync
```

### Sử dụng `pip`

Tất cả các ví dụ chạy trong cùng một môi trường **python 3.11**, vì vậy bạn nên tạo môi trường và cài đặt dependencies như sau:

```bash
# python -m venv .venv
# source .venv/bin/activate
pip install -r requirements.txt
```

### Google Colab

Nếu dùng **Google Colab** bạn sẽ cần cài đặt dependencies một cách linh hoạt dựa trên phần cứng bạn đang sử dụng. Như sau:

```bash
pip install -r transformers trl datasets huggingface_hub
```

## Tham gia và chia sẻ

Hãy chia sẻ khoá học này, để cùng nhau phát triển cộng đồng AI Việt Nam.

[![Biểu đồ sao của khoá học](https://api.star-history.com/svg?repos=huggingface/smol-course&type=Date)](https://star-history.com/#huggingface/smol-course&Date)
