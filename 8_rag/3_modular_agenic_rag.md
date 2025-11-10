# Modular (Agentic) RAG

## Why Modular RAG is Needed

**Vanilla RAG** faces two main challenges:

1. **Single retrieval step**: If the retrieved documents are irrelevant, the generated answer will be poor.
2. **Semantic mismatch**: The user’s query might differ in form from the document’s content, making semantic similarity-based retrieval suboptimal.

**Modular RAG** addresses these limitations by introducing an agent that can:

- **Formulate the query** to optimize document retrieval.
- **Critique and re-retrieve** if necessary, improving retrieval accuracy and ensuring better answers.

## Key Components of Modular RAG
### 1. **Module Components**

In a modular RAG system, module, tool, and agent are often used interchangeably, though they represent different levels of abstraction within the system. These components represent different parts of the system that work together to enhance functionality.

- **Search Module**: Expands retrieval by integrating data from various external sources like search engines, tabular data, and knowledge graphs, enhancing the relevance of context during retrieval. 
- **Memory Module**: Stores past interactions (queries and answers) for ongoing context awareness, supporting dynamic tasks and conversations.
- **Custom Function Tool Module**: Executes advanced workflows, such as database queries or system commands, allowing the agent to interact with external systems.
- **Code Module (Agent)**: Specializes in coding tasks like analysis, generation, refactoring, and testing, enabling the agent to handle software development tasks.

### 2. **Other Components**

- **Fusion**: Performs parallel retrieval on original and expanded queries, intelligently reranking and merging results for optimal context.
- **Routing**: Directs the next action based on the query, such as summarization or searching specific databases, ensuring appropriate responses.
- **Orchestration Agent**: Coordinates the flow of information between modules, optimizing the efficiency and effectiveness of the overall RAG system.

## AI Agent

AI agents are modular systems where the output of LLMs controls the workflow, enabling interaction with external tools, programs, or systems. They provide the necessary "agency" for LLMs to autonomously navigate tasks and processes. The agent's role is to translate LLM outputs into executable actions, bridging the gap between the language model and the real world.

AI agents bring an additional layer of intelligent orchestration, improving how different modules work together dynamically, rather than relying on static, predefined processes. Indeeds, agency in AI agents exists on a spectrum, with the LLM's control over the workflow increasing at each level:

| Agency Level | Description | Example Pattern |
| --- | --- | --- |
| ☆☆☆ | LLM output has no impact on program flow | Simple Processor (`process_llm_output(llm_response)`) |
| ★☆☆ | LLM output triggers an if/else switch | Router (`if llm_decision(): path_a() else: path_b()`) |
| ★★☆ | LLM output determines function execution | Tool Caller (`run_function(llm_chosen_tool, llm_chosen_args)`) |
| ★★★ | LLM output controls iteration | Multi-step Agent (`while llm_should_continue(memory): execute_next_step()`) |
| ★★★ | One agent starts another agentic workflow | Multi-Agent (`if llm_trigger(): execute_agent()`) |


![Agent](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-source-llms-as-agents/ReAct.png)


**When to Use Agents**

Agents are useful when flexibility is needed in the workflow. If tasks are too complex for predefined steps or criteria, an agent can adapt and determine the necessary actions. For simple tasks with a predictable workflow, agents may be unnecessary.


### `smolagent` Library

The **smolagent** library provides a simple yet powerful framework to build AI agents. While you can manually write code for simple agents for chaining or routing, more complex behaviors such as tool calling and multi-step agent workflows require predefined abstractions to work effectively. Here's why **smolagent** is helpful:

1. **Tool Calling**: When an agent needs to call a tool (e.g., fetching weather data), the output format from the LLM should be predefined, such as:  
   `Thought: I should call tool 'get_weather'. Action: get_weather(Paris).`  
   This ensures the LLM’s output can be parsed and executed by a system function.

2. **Multi-Step Agents**: If the agent’s output controls a loop (e.g., iterating over a series of tasks), a different prompt may be needed for each iteration based on memory. This requires integrating memory into the system.

Given these needs, **smolagent** provides essential building blocks that enable seamless orchestration:

- An LLM engine that powers the system
- A list of available tools the agent can use
- A parser that extracts tool calls from LLM output
- A memory system that stores relevant information
- A system prompt synced with the parser

Additionally, since agents are powered by LLMs, error logging, and retry mechanisms are essential for ensuring robustness and reliability. **smolagent** handles these elements, making it easier to build complex workflows that are reliable, flexible, and adaptive.


### Resources

https://huggingface.co/docs/smolagents/index
https://huggingface.co/docs/smolagents/examples/rag
https://huggingface.co/docs/smolagents/conceptual_guides/intro_agents