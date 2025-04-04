# Trợ lý Mã (Code Agents)

Trợ lý mã là các hệ thống tự động chuyên biệt có nhiệm vụ xử lý các tác vụ lập trình như phân tích, sinh mã, tái cấu trúc và kiểm thử. Những Trợ lý này tận dụng kiến thức chuyên ngành về các ngôn ngữ lập trình, hệ thống xây dựng và quản lý phiên bản để nâng cao quy trình phát triển phần mềm.

## Tại sao lại dùng Trợ lý Mã?

Trợ lý mã tăng tốc quá trình phát triển bằng cách tự động hóa các tác vụ lặp đi lặp lại trong khi vẫn duy trì chất lượng mã. Chúng xuất sắc trong việc sinh mã mẫu, thực hiện tái cấu trúc có hệ thống và nhận diện các vấn đề tiềm ẩn thông qua phân tích tĩnh. Các trợ lý kết hợp khả năng truy xuất để truy cập tài liệu và kho mã ngoài với việc gọi hàm để thực hiện các hành động cụ thể như tạo tệp hoặc chạy kiểm thử.

## Các thành phần cơ bản của Trợ lý Mã

Trợ lý mã được xây dựng trên các mô hình ngôn ngữ chuyên biệt được tinh chỉnh để hiểu mã. Những mô hình này được bổ sung với các công cụ phát triển như linters, formatters và compilers để tương tác với các môi trường thực tế. Thông qua các kỹ thuật truy xuất, các trợ lý duy trì nhận thức bối cảnh bằng cách truy cập tài liệu và lịch sử mã để phù hợp với các mô hình và tiêu chuẩn của tổ chức. Các hàm có mục đích hành động cho phép trợ lý thực hiện các tác vụ cụ thể như cam kết thay đổi hoặc khởi tạo yêu cầu gộp.

Trong ví dụ sau, chúng ta tạo một trợ lý mã có thể tìm kiếm web sử dụng DuckDuckGo, giống như trợ lý truy xuất mà chúng ta đã xây dựng trước đó.

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
```

Trong ví dụ tiếp theo, chúng ta tạo một trợ lý mã có thể lấy thời gian di chuyển giữa hai địa điểm. Ở đây, chúng ta sử dụng decorator `@tool` để định nghĩa một hàm tùy chỉnh có thể được sử dụng như một công cụ.

```python
from smolagents import CodeAgent, HfApiModel, tool

@tool
def get_travel_duration(start_location: str, destination_location: str, departure_time: Optional[int] = None) -> str:
    """Gets the travel time in car between two places.
    
    Args:
        start_location: the place from which you start your ride
        destination_location: the place of arrival
        departure_time: the departure time, provide only a `datetime.datetime` if you want to specify this
    """
    import googlemaps # All imports are placed within the function, to allow for sharing to Hub.
    import os

    gmaps = googlemaps.Client(os.getenv("GMAPS_API_KEY"))

    if departure_time is None:
        from datetime import datetime
        departure_time = datetime(2025, 1, 6, 11, 0)

    directions_result = gmaps.directions(
        start_location,
        destination_location,
        mode="transit",
        departure_time=departure_time
    )
    return directions_result[0]["legs"][0]["duration"]["text"]

agent = CodeAgent(tools=[get_travel_duration], model=HfApiModel(), additional_authorized_imports=["datetime"])

agent.run("Can you give me a nice one-day trip around Paris with a few locations and the times? Could be in the city or outside, but should fit in one day. I'm travelling only via public transportation.")
```

Những ví dụ trên chỉ là sự khởi đầu của những gì bạn có thể làm với trợ lý mã. Bạn có thể tìm hiểu thêm về cách xây dựng trợ lý mã trong [tài liệu smolagents](https://huggingface.co/docs/smolagents).

smolagents cung cấp một framework nhẹ để xây dựng các trợ lý mã, với triển khai lõi khoảng 1.000 dòng mã. Framework này chuyên về các trợ lý viết và thực thi các đoạn mã Python, cung cấp môi trường thực thi cách ly để đảm bảo an ninh. Nó hỗ trợ cả mô hình ngôn ngữ mã nguồn mở và bản quyền, giúp linh hoạt trong các môi trường phát triển khác nhau.

## Đọc thêm

- [Blog về smolagents](https://huggingface.co/blog/smolagents) - Giới thiệu về smolagents và các tương tác với mã nguồn.
- [smolagents: Xây dựng Trợ lý Tốt](https://huggingface.co/docs/smolagents/tutorials/building_good_agents) - Các thực hành tốt nhất cho các trợ lý đáng tin cậy.
- [Xây dựng Trợ lý Hiệu quả - Anthropic](https://www.anthropic.com/research/building-effective-agents) - Nguyên lý thiết kế trợ lý.