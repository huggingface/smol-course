# 智能体

AI 智能体（AI Agents）是一种自主系统，能够理解用户请求、将请求拆解为多个步骤，然后执行相应行动以完成任务。它们将语言模型与工具及外部功能相结合，从而与所处环境进行交互。本模章将讲解如何使用 [`smolagents`](https://github.com/huggingface/smolagents) 库构建高效的智能体，该库为创建强大的 AI 智能体提供了一个轻量级框架。

## 章节概览

高效的智能体通常具备三种关键能力。一是检索能力，智能体需要能够从各种信息来源获取和使用相关信息。二是函数调用，这使得智能体能够在所处环境中采取具体行动。最后是特定领域的知识和工具，这能让智能体可以执行诸如代码操作这类专业任务。

## 内容目录

### 1️⃣ [检索智能体](./retrieval_agents_cn.md)

检索型智能体将模型与知识库相结合。这些智能体能够从多个来源搜索并整合信息，借助向量数据库实现高效检索，并采用检索增强生成（Retrieval Augmented Generation 或 RAG）模式。它们擅长将网页搜索与自定义知识库相融合，同时通过记忆系统维持对话上下文。这部分教程主要讲解其实现策略，包括用于稳健信息检索的回退机制。

### 2️⃣ [代码智能体](./code_agents_cn.md)

代码智能体是为软件开发任务而设计的专门化自主系统。这些智能体擅长分析和生成代码、执行自动重构，并与开发工具相集成。这部分教程主要讲解如何构建以代码为核心的智能体，以及最佳实践。这些智能体能够理解编程语言、与构建系统（build systems）协作，并与版本控制系统交互，同时保持较高的代码质量标准。

### 3️⃣ [自定义函数](./custom_functions_cn.md)

自定义函数智能体通过专门的函数调用扩展了基本的人工智能能力。这部分教程将探讨如何设计模块化和可扩展的函数接口，使其直接与应用程序的逻辑相集成。你将学习在创建可靠的函数驱动的工作流时，如何实施适当的验证和错误处理。重点在于构建简单的系统，在该系统中智能体可以以可预测的方式与外部工具和服务进行交互。

### 练习

| 标题 | 概述 | 练习 | 链接 | Colab |
|-------|-------------|----------|------|-------|
| 构建科研智能体 | 构建一个借助检索和自定义函数可以执行科研任务的智能体 | 🐢 构建简单的 RAG 智能体 <br> 🐕 加入自定义的搜索函数 <br> 🦁 创建完整的科研助手 | [Notebook](./notebooks/agents_cn.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/8_agents/notebooks/agents.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## Resources

- [smolagents Documentation](https://huggingface.co/docs/smolagents) - smolagents 代码库的官方文档
- [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - 智能体架构方面的研究性论文
- [Agent Guidelines](https://huggingface.co/docs/smolagents/tutorials/building_good_agents) - 构建可靠智能体的最佳实践
- [LangChain Agents](https://python.langchain.com/docs/how_to/#agents) - 智能体实现的一些示例
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) - 理解 LLM 中的函数调用
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/) - 高效 RAG 实现指南
