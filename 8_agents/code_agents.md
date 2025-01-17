# Code Agents

Code agents are specialized autonomous systems that handle coding tasks like analysis, generation, refactoring, and testing. These agents leverage domain knowledge about programming languages, build systems, and version control to enhance software development workflows.

## Why Code Agents?

Code agents accelerate development by automating repetitive tasks while maintaining code quality. They excel at generating boilerplate code, performing systematic refactoring, and identifying potential issues through static analysis. The agents combine retrieval capabilities to access external documentation and repositories with function calling to execute concrete actions like creating files or running tests.

## Building Blocks of a Code Agent

Code agents are built on specialized language models fine-tuned for code understanding. These models are augmented with development tools like linters, formatters, and compilers to interact with real-world environments. Through retrieval techniques, agents maintain contextual awareness by accessing documentation and code histories to align with organizational patterns and standards. Action-oriented functions enable agents to perform concrete tasks such as committing changes or initiating merge requests.

In the following example, we create a code agent that can search the web using DuckDuckGo much like the retrieval agent we built earlier.

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
```

In the following example, we create a code agent that can get the travel time between two locations. Here, we use the `@tool` decorator to define a custom function that can be used as a tool.

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

These examples are just the beginning of what you can do with code agents. You can learn more about how to build code agents in the [smolagents documentation](https://huggingface.co/docs/smolagents).

smolagents provides a lightweight framework for building code agents, with a core implementation of approximately 1,000 lines of code. The framework specializes in agents that write and execute Python code snippets, offering sandboxed execution for security. It supports both open-source and proprietary language models, making it adaptable to various development environments.

## Further Reading

- [smolagents Blog](https://huggingface.co/blog/smolagents) - Introduction to smolagents and code interactions
- [smolagents: Building Good Agents](https://huggingface.co/docs/smolagents/tutorials/building_good_agents) - Best practices for reliable agents
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents) - Agent design principles
