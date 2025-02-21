# 代码智能体

代码智能体是专门的自主系统，可处理诸如分析、生成、重构和测试之类的代码任务。这些智能体利用关于编程语言、构建系统（build systems）和版本控制的领域知识来增强软件开发工作流程。

## 为什么我们需要代码智能体？

代码智能体通过将重复性任务自动化，同时保持代码质量，从而加速开发过程。它们擅长生成样板代码、进行系统性重构，并通过静态分析识别潜在问题。这些智能体将检索能力（访问外部文档和代码库）与函数调用相结合，以执行创建文件或运行测试等具体操作。

## 代码智能体的组成

代码智能体建立在为代码理解而微调过的专门语言模型之上。这些模型辅以诸如代码检查工具、格式化工具和编译器等开发工具，以便与现实世界环境进行交互。通过检索技术，智能体通过访问文档和代码历史记录来保持上下文感知，以符合组织的模式和标准。以行动为导向的函数使智能体能够执行具体任务，例如提交变更或发起合并请求。

在以下示例中，我们创建一个代码智能体，它可以像我们之前构建的检索智能体那样使用 DuckDuckGo 进行网络搜索。

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
```

在以下示例中，我们创建了一个代码智能体，它能够获取两个位置之间的运动时间。这里，我们使用 `@tool` 装饰器来定义一个可用作工具的自定义函数。

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

当然，这两个例子仅仅是你开发代码智能体的开始。你需要阅读 [smolagents 文档](https://huggingface.co/docs/smolagents)来了解更多构建代码智能体的方法。

`smolagents` 为构建代码智能体提供了一个轻量级的框架，其核心代码实现仅有 1000 行左右。该框架专注于构建能够编写和执行 Python 代码片段的智能体，并提供沙盒式执行环境以确保安全性。它同时支持开源和专有语言模型，使其能够适配各种开发环境。

## 延伸阅读

- [smolagents Blog](https://huggingface.co/blog/smolagents) - 介绍 smolagents 的博客
- [smolagents: Building Good Agents](https://huggingface.co/docs/smolagents/tutorials/building_good_agents) - 构建可靠智能体的最佳实践
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents) - 智能体设计原则
