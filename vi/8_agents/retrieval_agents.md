# Xây dựng Hệ thống Trợ lý RAG

Trợ lý RAG (Retrieval Augmented Generation) kết hợp sức mạnh của các trợ lý tự động với khả năng truy xuất kiến thức. Trong khi các hệ thống RAG truyền thống chỉ sử dụng một Mô hình Ngôn ngữ Lớn (LLM) để trả lời các truy vấn dựa trên thông tin đã truy xuất, trợ lý RAG nâng cao điều này bằng cách cho phép hệ thống tự động điều khiển quá trình truy xuất và phản hồi của chính nó một cách thông minh.

RAG truyền thống có những hạn chế quan trọng - nó chỉ thực hiện một bước truy xuất đơn lẻ và dựa vào sự tương đồng ngữ nghĩa trực tiếp với truy vấn người dùng, điều này có thể bỏ lỡ thông tin liên quan. Trợ lý RAG giải quyết những thách thức này bằng cách trao quyền cho trợ lý để xây dựng các truy vấn tìm kiếm của riêng mình, đánh giá kết quả và thực hiện nhiều bước truy xuất khi cần thiết.

## Truy xuất cơ bản với DuckDuckGo

Chúng ta bắt đầu bằng việc xây dựng một trợ lý đơn giản có thể tìm kiếm web sử dụng DuckDuckGo. Trợ lý này sẽ có khả năng trả lời các câu hỏi bằng cách truy xuất thông tin liên quan và tổng hợp các phản hồi.

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

# Khởi tạo công cụ tìm kiếm
search_tool = DuckDuckGoSearchTool()

# Khởi tạo mô hình
model = HfApiModel()

agent = CodeAgent(
    model = model,
    tools=[search_tool]
)

# Ví dụ sử dụng
response = agent.run(
    "What are the latest developments in fusion energy?"
)
print(response)
```

Trợ lý sẽ:
1. Phân tích truy vấn để xác định thông tin cần thiết
2. Sử dụng DuckDuckGo để tìm kiếm nội dung liên quan
3. Tổng hợp thông tin đã truy xuất thành một phản hồi mạch lạc
4. Lưu trữ tương tác trong bộ nhớ để tham chiếu trong tương lai

## Công cụ Cơ sở Kiến thức Tùy chỉnh

Đối với các ứng dụng chuyên ngành, chúng ta thường muốn kết hợp tìm kiếm web với cơ sở kiến thức của riêng mình. Hãy tạo một công cụ tùy chỉnh có thể truy vấn cơ sở dữ liệu vector của tài liệu kỹ thuật.

```python
from smolagents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = "Sử dụng tìm kiếm ngữ nghĩa để truy xuất các phần tài liệu transformers có thể liên quan nhất để trả lời truy vấn của bạn."
    inputs = {
        "query": {
            "type": "string",
            "description": "Truy vấn để thực hiện. Truy vấn này nên gần về ngữ nghĩa với các tài liệu mục tiêu. Sử dụng dạng khẳng định thay vì câu hỏi.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=10
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Truy vấn tìm kiếm của bạn phải là chuỗi"

        docs = self.retriever.invoke(
            query,
        )
        return "\nTài liệu truy xuất:\n" + "".join(
            [
                f"\n\n===== Tài liệu {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

retriever_tool = RetrieverTool(docs_processed)
```

Trợ lý nâng cao này có thể:
1. Kiểm tra tài liệu để tìm thông tin liên quan trước
2. Tìm kiếm web nếu cần
3. Kết hợp thông tin từ cả hai nguồn
4. Duy trì bối cảnh cuộc trò chuyện thông qua bộ nhớ

## Nâng cao Khả năng Truy xuất

Khi xây dựng các hệ thống Trợ lý RAG, trợ lý có thể sử dụng các chiến lược phức tạp như:

1. **Cải thiện Truy vấn** - Thay vì sử dụng truy vấn người dùng ban đầu, trợ lý có thể tạo ra các thuật ngữ tìm kiếm tối ưu hơn, phù hợp hơn với các tài liệu mục tiêu.
2. **Truy xuất Đa bước** - Trợ lý có thể thực hiện nhiều tìm kiếm, sử dụng kết quả ban đầu để cải thiện các truy vấn tiếp theo.
3. **Tích hợp Nguồn** - Thông tin có thể được kết hợp từ nhiều nguồn như tìm kiếm web và tài liệu nội bộ.
4. **Xác thực Kết quả** - Nội dung truy xuất có thể được phân tích để xác định sự phù hợp và chính xác trước khi được đưa vào phản hồi.

Các hệ thống Trợ lý RAG hiệu quả đòi hỏi phải xem xét cẩn thận nhiều yếu tố quan trọng. Trợ lý nên lựa chọn công cụ phù hợp dựa trên loại truy vấn và bối cảnh. Các hệ thống bộ nhớ giúp duy trì lịch sử cuộc trò chuyện và tránh các truy xuất lặp lại. Các chiến lược dự phòng giúp đảm bảo hệ thống vẫn có thể cung cấp giá trị ngay cả khi các phương pháp truy xuất chính không thành công. Thêm vào đó, việc triển khai các bước xác thực giúp đảm bảo độ chính xác và sự phù hợp của thông tin truy xuất.

```python
import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
docs_processed = text_splitter.split_documents(source_docs)
```

## Các Bước Tiếp Theo

⏩ Xem qua module [Code Agents](./code_agents.md) để học cách xây dựng các trợ lý có thể thao tác với mã nguồn.