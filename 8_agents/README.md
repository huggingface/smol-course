# Agents

AI Agents are autonomous systems that can understand user requests, break them down into steps, and execute actions to accomplish tasks. They combine language models with tools and external functions to interact with their environment. This module covers how to build effective agents using the [`smolagents`](https://github.com/huggingface/smolagents) library, which provides a lightweight framework for creating capable AI agents.

## Module Overview

Building effective agents requires understanding three key components. First, retrieval capabilities allow agents to access and use relevant information from various sources. Second, function calling enables agents to take concrete actions in their environment. Finally, domain-specific knowledge and tooling equip agents for specialized tasks like code manipulation.

## Contents

### 1️⃣ [Retrieval Agents](./retrieval_agents.md)

Retrieval agents combine models with knowledge bases. These agents can search and synthesize information from multiple sources, leveraging vector stores for efficient retrieval and implementing RAG (Retrieval Augmented Generation) patterns. They are great at combining web search with custom knowledge bases while maintaining conversation context through memory systems. The module covers implementation strategies including fallback mechanisms for robust information retrieval.

### 2️⃣ [Code Agents](./code_agents.md)

Code agents are specialized autonomous systems designed for software development tasks. These agents excel at analyzing and generating code, performing automated refactoring, and integrating with development tools. The module covers best practices for building code-focused agents that can understand programming languages, work with build systems, and interact with version control while maintaining high code quality standards.

### 3️⃣ [Custom Functions](./custom_functions.md)

Custom function agents extend basic AI capabilities through specialized function calls. This module explores how to design modular and extensible function interfaces that integrate directly with your application's logic. You'll learn to implement proper validation and error handling while creating reliable function-driven workflows. The focus is on building simple systems where agents can predictably interact with external tools and services.

### Exercise Notebooks

| Title | Description | Exercise | Link | Colab |
|-------|-------------|----------|------|-------|
| Building a Research Agent | Create an agent that can perform research tasks using retrieval and custom functions | 🐢 Build a simple RAG agent <br> 🐕 Add custom search functions <br> 🦁 Create a full research assistant | [Notebook](./notebooks/agents.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/8_agents/notebooks/agents.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## Resources

- [smolagents Documentation](https://huggingface.co/docs/smolagents) - Official docs for the smolagents library
- [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - Research paper on agent architectures
- [Agent Guidelines](https://huggingface.co/docs/smolagents/tutorials/building_good_agents) - Best practices for building reliable agents
- [LangChain Agents](https://python.langchain.com/docs/how_to/#agents) - Additional examples of agent implementations
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) - Understanding function calling in LLMs
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/) - Guide to implementing effective RAG
