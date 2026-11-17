# Suy luận

Suy luận là quá trình sử dụng mô hình ngôn ngữ đã được huấn luyện để tạo ra các dự đoán hoặc phản hồi. Mặc dù suy luận có vẻ đơn giản, nhưng triển khai các mô hình một cách hiệu quả ở quy mô lớn đòi hỏi phải xem xét kỹ lưỡng nhiều yếu tố như hiệu suất, chi phí và độ tin cậy. Các Mô hình Ngôn ngữ Lớn (LLMs) mang lại những thách thức đặc biệt do kích thước và yêu cầu tính toán của chúng.

Chúng ta sẽ khám phá cả hai phương pháp đơn giản và sẵn sàng cho sản xuất bằng cách sử dụng thư viện [`transformers`](https://huggingface.co/docs/transformers/index) và [`text-generation-inference`](https://github.com/huggingface/text-generation-inference), hai framework phổ biến cho suy luận LLM. Đối với các triển khai sản xuất, chúng ta sẽ tập trung vào Suy luận Sinh văn Bằng Văn bản (TGI), cung cấp khả năng phục vụ tối ưu.

## Tổng quan về Module

Suy luận LLM có thể được phân thành hai phương pháp chính: suy luận dựa trên pipeline đơn giản cho phát triển và thử nghiệm, và các giải pháp phục vụ tối ưu cho triển khai sản xuất. Chúng ta sẽ đề cập đến cả hai phương pháp, bắt đầu với phương pháp pipeline đơn giản và tiến tới các giải pháp sẵn sàng cho sản xuất.

## Nội dung

### 1. [Suy luận Pipeline Cơ bản](./pipeline_inference.md)

Học cách sử dụng pipeline Hugging Face Transformers cho suy luận cơ bản. Chúng ta sẽ tìm hiểu cách thiết lập pipeline, cấu hình các tham số sinh văn và các thực hành tốt nhất cho phát triển cục bộ. Phương pháp pipeline là lựa chọn hoàn hảo cho việc tạo mẫu và các ứng dụng quy mô nhỏ. [Bắt đầu học](./pipeline_inference.md).

### 2. [Suy luận Sản xuất với TGI](./tgi_inference.md)

Học cách triển khai mô hình cho sản xuất bằng cách sử dụng Suy luận Sinh văn Bằng Văn bản. Chúng ta sẽ khám phá các kỹ thuật phục vụ tối ưu, chiến lược batching và giải pháp giám sát. TGI cung cấp các tính năng sẵn sàng cho sản xuất như kiểm tra tình trạng, số liệu và các tùy chọn triển khai Docker. [Bắt đầu học](./text_generation_inference.md).

### Sổ tay Bài tập

| Tiêu đề           | Mô tả                                     | Bài tập                                                                                                      | Liên kết                                               | Colab                                                                                                                     |
| ----------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| Suy luận Pipeline | Suy luận cơ bản với pipeline transformers | 🐢 Thiết lập pipeline cơ bản <br> 🐕 Cấu hình tham số sinh văn <br> 🦁 Tạo máy chủ web đơn giản           | [Liên kết](./notebooks/basic_pipeline_inference.ipynb) | [Colab](https://githubtocolab.com/huggingface/smol-course/tree/main/7_inference/notebooks/basic_pipeline_inference.ipynb) |
| Triển khai TGI    | Triển khai sản xuất với TGI               | 🐢 Triển khai mô hình với TGI <br> 🐕 Cấu hình tối ưu hóa hiệu suất <br> 🦁 Thiết lập giám sát và mở rộng | [Liên kết](./notebooks/tgi_deployment.ipynb)           | [Colab](https://githubtocolab.com/huggingface/smol-course/tree/main/7_inference/notebooks/tgi_deployment.ipynb)           |

## Tài nguyên

- [Hướng dẫn Pipeline Hugging Face.](https://huggingface.co/docs/transformers/en/pipeline_tutorial)
- [Tài liệu Suy luận Sinh văn Bằng Văn bản.](https://huggingface.co/docs/text-generation-inference/en/index)
- [Hướng dẫn Pipeline WebServer.](https://huggingface.co/docs/transformers/en/pipeline_tutorial#using-pipelines-for-a-webserver)
- [Kho lưu trữ TGI trên GitHub.](https://github.com/huggingface/text-generation-inference)
- [Tài liệu Triển khai Mô hình Hugging Face.](https://huggingface.co/docs/inference-endpoints/index)
- [vLLM: Phục vụ LLM Tốc độ Cao.](https://github.com/vllm-project/vllm)
- [Tối ưu hóa Suy luận Transformer.](https://huggingface.co/blog/optimize-transformer-inference)