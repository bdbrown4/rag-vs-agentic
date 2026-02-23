"""
Agentic Retrieval using a ReAct agent.

Flow:
    User query → LLM reasons → picks a tool → observes result → reasons again → ...repeat... → final answer

Key differences from RAG:
- The agent DECIDES when and what to retrieve (not hardcoded)
- It can refine queries based on intermediate results
- It can chain multiple tool calls (search → get full doc → search again)
- It can pull from heterogeneous sources (vector store + live GitHub API)
- Every step produces a visible thought → action → observation trace
"""

import time
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate

from agentic.tools import ALL_TOOLS


@dataclass
class AgenticResult:
    """Result from the agentic pipeline, including trace info for comparison."""
    question: str
    answer: str
    llm_calls: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    total_tokens: int = 0
    latency_seconds: float = 0.0
    steps: list[str] = field(default_factory=list)


AGENT_SYSTEM_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant answering questions about a developer's GitHub portfolio.
You have access to tools that let you search a README knowledge base, get full documents,
list available repos, and fetch live data from GitHub.

IMPORTANT: Think step-by-step about what information you need. You can:
1. Search for relevant content across all repos
2. Get the full README of a specific repo for deeper context
3. List all available repos to understand the full portfolio
4. Fetch live GitHub data for real-time info (stars, issues, recent activity)

Use multiple tools if needed to build a complete answer. If your first search
doesn't find what you need, try rephrasing or looking at the full document.

TOOLS:
------
You have access to the following tools:

{tools}

To use a tool, use the following format:

Thought: I need to think about what to do next.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When you have enough information to answer, respond with:

Thought: I now have enough information to answer.
Final Answer: your complete answer here

Begin!

Question: {input}

{agent_scratchpad}"""
)


def run_agentic_pipeline(
    question: str,
    model: str = "gpt-4o",
    max_iterations: int = 8,
    verbose: bool = True,
) -> AgenticResult:
    """
    Execute the agentic retrieval pipeline.

    Args:
        question: User's question.
        model: OpenAI model name.
        max_iterations: Max reasoning/tool loops before forcing a final answer.
        verbose: Whether to print intermediate steps.

    Returns:
        AgenticResult with answer, tool call trace, and performance metrics.
    """
    start = time.time()
    steps = []
    tool_calls = []

    llm = ChatOpenAI(model=model, temperature=0)

    agent = create_react_agent(
        llm=llm,
        tools=ALL_TOOLS,
        prompt=AGENT_SYSTEM_PROMPT,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=verbose,
        max_iterations=max_iterations,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

    # Run the agent
    result = executor.invoke({"input": question})

    # Extract trace information
    answer = result.get("output", "No answer produced.")
    intermediate_steps = result.get("intermediate_steps", [])

    llm_calls = 0
    for action, observation in intermediate_steps:
        llm_calls += 1
        tool_name = action.tool
        tool_input = action.tool_input
        steps.append(f"THOUGHT: Agent decided to use '{tool_name}'")
        steps.append(f"ACTION: {tool_name}({tool_input})")
        steps.append(f"OBSERVATION: {str(observation)[:200]}...")

        tool_calls.append({
            "tool": tool_name,
            "input": tool_input,
            "output_preview": str(observation)[:200],
        })

    # +1 for the final answer generation
    llm_calls += 1
    steps.append(f"FINAL ANSWER: Generated response after {len(intermediate_steps)} tool calls")

    elapsed = time.time() - start

    return AgenticResult(
        question=question,
        answer=answer,
        llm_calls=llm_calls,
        tool_calls=tool_calls,
        latency_seconds=round(elapsed, 2),
        steps=steps,
    )
