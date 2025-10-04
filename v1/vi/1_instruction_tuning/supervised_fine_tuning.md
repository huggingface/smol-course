# Tinh chỉnh có giám sát (Supervised Fine-Tuning)

Tinh chỉnh có giám sát (SFT) là một quá trình cốt lõi để điều chỉnh *pre-trained models* cho các tác vụ hoặc lĩnh vực cụ thể. Mặc dù các mô hình đã *pre-trained* có các khả năng tổng quát ấn tượng  ,chúng thường cần được tùy chỉnh để đạt hiệu suất cao trong các trường hợp sử dụng cụ thể. SFT thu hẹp khoảng cách này bằng cách huấn luyện thêm mô hình trên các tập dữ liệu được tuyển chọn kỹ lưỡng cùng với các ví dụ đã được con người đánh giá.

## Tìm hiểu thêm về huấn luyện có giám sát

Về cốt lõi, SFT là về việc dạy một mô hình đã *pre-trained* thực hiện các tác vụ cụ thể thông qua các mẫu của các *token* đã được gán nhãn. Quá trình này bao gồm việc cho mô hình học nhiều mẫu về các hành vi đầu vào-đầu ra mong muốn, cho phép nó học các mẫu cụ thể cho trường hợp sử dụng của bạn.

SFT hiệu quả vì nó sử dụng kiến thức nền tảng thu được trong quá trình *pre-training* trong khi điều chỉnh hành vi của mô hình để phù hợp với nhu cầu cụ thể của bạn.

## Khi nào nên sử dụng huấn luyện có giám sát

Quyết định sử dụng SFT thường phụ thuộc vào khoảng cách giữa khả năng hiện tại của mô hình và yêu cầu cụ thể của bạn. SFT trở nên đặc biệt có giá trị khi bạn cần kiểm soát chính xác đầu ra của mô hình hoặc khi làm việc trong các lĩnh vực chuyên biệt.

Ví dụ, nếu bạn đang phát triển một ứng dụng dịch vụ khách hàng, bạn có thể muốn mô hình của mình liên tục tuân theo hướng dẫn của công ty và xử lý các truy vấn kỹ thuật theo cách chuẩn hóa. Tương tự, trong các ứng dụng y tế hoặc pháp lý, độ chính xác và tuân thủ thuật ngữ chuyên ngành trở nên cực kỳ quan trọng. Trong những trường hợp này, SFT có thể giúp điều chỉnh phản hồi của mô hình phù hợp với tiêu chuẩn chuyên môn và chuyên môn trong lĩnh vực.

## Quy trình huấn luyện

Quy trình huấn luyện có giám sát bao gồm việc huấn luyện trọng số mô hình trên một tập dữ liệu theo tác vụ cụ thể.

Đầu tiên, bạn cần chuẩn bị hoặc lựa chọn một tập dữ liệu đại diện cho tác vụ mục tiêu của bạn. Tập dữ liệu này nên bao gồm các mẫu đa dạng bao quát phạm vi các tình huống mà mô hình của bạn sẽ gặp phải. Chất lượng của dữ liệu này rất quan trọng - mỗi mẫu nên thể hiện loại đầu ra mà bạn muốn mô hình của mình tạo ra. Tiếp theo là giai đoạn huấn luyện thực tế, nơi bạn sẽ sử dụng các framework có sẵn như `transformers` và `trl` của Hugging Face để huấn luyện mô hình trên tập dữ liệu của bạn.

Trong suốt quá trình này, việc đánh giá liên tục là thiết yếu. Bạn sẽ muốn theo dõi hiệu suất của mô hình trên một tập đánh giá (validation set) để đảm bảo nó đang học các hành vi mong muốn mà không mất đi khả năng tổng quát. Trong [bài 4](./4_evaluation), chúng ta sẽ tìm hiểu cách đánh giá mô hình đã được huấn luyện.

## Vai trò của huấn luyện có giám sát trong điều chỉnh theo sự uy tiên

SFT đóng vai trò nền tảng trong việc điều chỉnh các mô hình ngôn ngữ theo ưu tiên của con người. Các kỹ thuật như Học tăng cường từ phản hồi của con người (Reinforcement Learning from Human Feedback - RLHF) và Tối ưu hóa sở thích trực tiếp (Direct Preference Optimization - DPO) dựa vào SFT để xây dựng mức độ hiểu biết cơ bản về tác vụ trước khi tiếp tục điều chỉnh phản hồi của mô hình sao cho phù hợp với kết quả mong muốn. Các mô hình đã *pre-trained*, mặc dù có khả năng ngôn ngữ tổng quát, có thể không phải lúc nào cũng tạo ra đầu ra phù hợp ưu tiên của con người. SFT thu hẹp khoảng cách này bằng cách đưa vào dữ liệu và hướng dẫn theo lĩnh vực cụ thể, cải thiện khả năng của mô hình trong việc tạo ra phản hồi phù hợp hơn với kỳ vọng của con người.

## Huấn luyện có giám sát với Transformer Reinforcement Learning

Một thư viện quan trọng cho SFT đó là Transformer Reinforcement Learning (TRL). TRL là một bộ công cụ được sử dụng để huấn luyện các mô hình ngôn ngữ transformer bằng học tăng cường (RL).

Được xây dựng trên thư viện Transformers của Hugging Face, TRL cho phép người dùng trực tiếp tải các mô hình ngôn ngữ đã được *pre-trained* và hỗ trợ hầu hết các kiến trúc decoder và encoder-decoder. Thư viện này tạo điều kiện cho các quy trình chính của RL được sử dụng trong mô hình hóa ngôn ngữ, bao gồm Supervised Fine-Tuning (SFT), Reward Modeling (RM), Proximal Policy Optimization (PPO), và Direct Preference Optimization (DPO). Chúng ta sẽ sử dụng TRL trong nhiều bài học trong khoá học này.

# Các bước tiếp theo

Hãy thử các hướng dẫn sau để có tìm hiểu các ví dụ SFT thông qua TRL:

⏭️ [Hướng dẫn Địng dạng Chat](./notebooks/chat_templates_example.ipynb)

⏭️ [Hướng dẫn Huấn luyện có giám sát](./notebooks/supervised_fine_tuning_tutorial.ipynb)