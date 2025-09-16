# Đánh giá mô hình

Đánh giá là bước quan trọng trong việc phát triển và triển khai các mô hình ngôn ngữ. Nó giúp chúng ta hiểu được mô hình hoạt động tốt như thế nào qua các khả năng khác nhau và xác định những lĩnh vực cần cải thiện. Chương này bao gồm các phương pháp đánh giá (benchmark) tiêu chuẩn và các phương pháp đánh giá theo lĩnh vực cụ thể để đánh giá toàn diện mô hình của bạn.

Chúng ta sẽ sử dụng [`lighteval`](https://github.com/huggingface/lighteval), một thư viện chuyên được sử dụng để đánh giá mô hình ngôn ngữ được phát triển bởi Hugging Face. Thư viện này được thiết kế tối giản giúp người sử dụng dễ dàng tiếp cận và được tích hợp để phù hợp với hệ sinh thái của Hugging Face. Để tìm hiểu sâu hơn về các khái niệm và thực hành về đánh giá mô hình ngôn ngữ, hãy tham khảo [hướng dẫn](https://github.com/huggingface/evaluation-guidebook) về đánh giá.

## Tổng quan chương học

Một quy trình đánh giá mô hình thường bao gồm nhiều khía cạnh về hiệu năng của mô hình. Chúng ta đánh giá các khả năng theo tác vụ cụ thể như *trả lời câu hỏi* và *tóm tắt* để hiểu mô hình xử lý các loại vấn đề khác nhau tốt đến đâu. Chúng ta đo lường chất lượng đầu ra thông qua các yếu tố như *tính mạch lạc* và *độ chính xác*. Đánh giá về *độ an toàn* giúp xác định các đầu ra của mô hình có chứa các hành vi khuyển khích thực hiện hành đông độc hại hoặc định kiến tiềm ẩn. Cuối cùng, kiểm tra *chuyên môn* về lĩnh vực xác minh kiến thức chuyên biệt của mô hình trong lĩnh vực mục tiêu của bạn.

## Nội dung

### 1️⃣ [Đánh giá mô hình tự động](./automatic_benchmarks.md)

Học cách đánh giá mô hình của bạn bằng các phương pháp đánh giá và số liệu chuẩn hoá. Chúng ta sẽ khám phá các phương pháp đánh giá phổ biến như `MMLU` và `TruthfulQA`, hiểu các chỉ số đánh giá và cài đặt quan trọng, đồng thời đề cập đến các thực hành tốt nhất cho việc đánh giá có thể tái tạo lại.

### 2️⃣ [Đánh giá theo yêu cầu đặc biệt](./custom_evaluation.md)
Khám phá cách tạo quy trình (pipeline) đánh giá phù hợp với trường hợp sử dụng cụ thể của bạn. Chúng ta sẽ hướng dẫn thiết kế các tác vụ đánh giá theo yêu cầu (custom), triển khai các số liệu chuyên biệt và xây dựng tập dữ liệu đánh giá phù hợp với yêu cầu của bạn.

### 3️⃣ [Dự án đánh giá theo lĩnh vực cụ thể](./project/README.md)
Theo dõi một ví dụ hoàn chỉnh về việc xây dựng quy trình đánh giá cho lĩnh vực cụ thể. Bạn sẽ học cách tạo tập dữ liệu đánh giá, sử dụng `Argilla` để gán nhãn dữ liệu, tạo tập dữ liệu chuẩn hoá và đánh giá mô hình bằng `LightEval`.

## Tài liệu tham khảo

- [Hướng dẫn đánh giá](https://github.com/huggingface/evaluation-guidebook) - Hướng dẫn toàn diện về đánh giá LLM
- [Tài liệu LightEval](https://github.com/huggingface/lighteval) - Tài liệu chính thức cho thư viện LightEval 
- [Tài liệu Argilla](https://docs.argilla.io) - Tìm hiểu về nền tảng gán nhãn Argilla
- [Paper MMLU](https://arxiv.org/abs/2009.03300) - Paper mô tả benchmark MMLU
- [Tạo tác vụ theo yêu cầu](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [Tạo tiêu chuẩn đánh giá theo yêu cầu](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- [Sử dụng số liệu có sẵn](https://github.com/huggingface/lighteval/wiki/Metric-List)