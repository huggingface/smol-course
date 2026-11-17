# Trợ lý AI

Trợ lý AI là các hệ thống tự động có khả năng hiểu các yêu cầu của người dùng, phân tích chúng thành các bước và thực hiện các hành động để hoàn thành nhiệm vụ. Các trợ lý này kết hợp mô hình ngôn ngữ với công cụ và chức năng bên ngoài để tương tác với môi trường. Module này hướng dẫn cách xây dựng các trợ lý hiệu quả sử dụng thư viện [`smolagents`](https://github.com/huggingface/smolagents), cung cấp một framework nhẹ để tạo ra các trợ lý AI có khả năng.

## Tổng quan về Module

Việc xây dựng các trợ lý hiệu quả đòi hỏi phải hiểu ba thành phần chính. Đầu tiên, khả năng truy xuất cho phép trợ lý truy cập và sử dụng thông tin từ nhiều nguồn khác nhau. Thứ hai, gọi chức năng cho phép trợ lý thực hiện các hành động cụ thể trong môi trường của nó. Cuối cùng, kiến thức chuyên ngành và các công cụ trang bị cho trợ lý các tác vụ chuyên biệt như thao tác mã nguồn.

## Nội dung

### 1️⃣ [Trợ lý Truy xuất](./retrieval_agents.md)

Trợ lý truy xuất kết hợp các mô hình với các cơ sở kiến thức. Những trợ lý này có thể tìm kiếm và tổng hợp thông tin từ nhiều nguồn, tận dụng các kho lưu trữ vector để truy xuất hiệu quả và triển khai các mẫu RAG (Retrieval Augmented Generation). Chúng rất giỏi trong việc kết hợp tìm kiếm web với cơ sở kiến thức tùy chỉnh, đồng thời duy trì bối cảnh cuộc trò chuyện thông qua các hệ thống bộ nhớ. Module này bao gồm các chiến lược triển khai, bao gồm các cơ chế dự phòng để truy xuất thông tin đáng tin cậy.

### 2️⃣ [Trợ lý Mã nguồn](./code_agents.md)

Trợ lý mã nguồn là các hệ thống tự động chuyên biệt được thiết kế cho các tác vụ phát triển phần mềm. Những trợ lý này xuất sắc trong việc phân tích và sinh mã nguồn, thực hiện tái cấu trúc tự động và tích hợp với các công cụ phát triển. Module này cung cấp các thực hành tốt nhất để xây dựng các trợ lý tập trung vào mã nguồn có thể hiểu ngôn ngữ lập trình, làm việc với hệ thống xây dựng và tương tác với quản lý phiên bản trong khi duy trì các tiêu chuẩn chất lượng mã cao.

### 3️⃣ [Chức năng Tùy chỉnh](./custom_functions.md)

Trợ lý chức năng tùy chỉnh mở rộng các khả năng AI cơ bản thông qua các lời gọi chức năng chuyên biệt. Module này khám phá cách thiết kế các giao diện chức năng mô-đun và có thể mở rộng để tích hợp trực tiếp với logic ứng dụng của bạn. Bạn sẽ học cách triển khai xác thực và xử lý lỗi thích hợp trong khi tạo ra các quy trình làm việc dựa trên chức năng đáng tin cậy. Mục tiêu là xây dựng các hệ thống đơn giản, nơi các trợ lý có thể tương tác một cách dự đoán với các công cụ và dịch vụ bên ngoài.

### Sổ tay Bài tập

| Tiêu đề | Mô tả | Bài tập | Liên kết | Colab |
|---------|-------|---------|---------|-------|
| Xây dựng Trợ lý Nghiên cứu | Tạo một trợ lý có thể thực hiện các nhiệm vụ nghiên cứu sử dụng truy xuất và chức năng tùy chỉnh | 🐢 Xây dựng một trợ lý RAG đơn giản <br> 🐕 Thêm các chức năng tìm kiếm tùy chỉnh <br> 🦁 Tạo trợ lý nghiên cứu đầy đủ | [Sổ tay](./notebooks/agents.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/8_agents/notebooks/agents.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Mở trong Colab"/></a> |

## Tài nguyên

- [Tài liệu smolagents](https://huggingface.co/docs/smolagents) - Tài liệu chính thức cho thư viện smolagents.
- [Xây dựng Trợ lý Hiệu quả](https://www.anthropic.com/research/building-effective-agents) - Bài nghiên cứu về kiến trúc trợ lý.
- [Hướng dẫn Trợ lý](https://huggingface.co/docs/smolagents/tutorials/building_good_agents) - Các thực hành tốt nhất để xây dựng các trợ lý đáng tin cậy.
- [Trợ lý LangChain](https://python.langchain.com/docs/how_to/#agents) - Các ví dụ bổ sung về triển khai trợ lý.
- [Hướng dẫn Gọi Chức năng](https://platform.openai.com/docs/guides/function-calling) - Hiểu rõ về gọi chức năng trong LLM.
- [Thực hành RAG Tốt nhất](https://www.pinecone.io/learn/retrieval-augmented-generation/) - Hướng dẫn triển khai RAG hiệu quả.