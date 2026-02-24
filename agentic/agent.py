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
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from agentic.tools import ALL_TOOLS
from langchain_community.callbacks import get_openai_callback
from shared.metrics import estimate_cost


@dataclass
class AgenticResult:
    """Result from the agentic pipeline, including trace info for comparison."""
    question: str
    answer: str
    llm_calls: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    steps: list[str] = field(default_factory=list)


SYSTEM_PROMPT = """You are a helpful assistant answering questions about a developer's GitHub portfolio.
You have access to tools that let you search a README knowledge base, get full documents,
list available repos, and fetch live data from GitHub.

IMPORTANT: Think step-by-step about what information you need. You can:
1. Search for relevant content across all repos
2. Get the full README of a specific repo for deeper context
3. List all available repos to understand the full portfolio
4. Fetch live GitHub data for real-time info (stars, issues, recent activity)

Use multiple tools if needed to build a complete answer. If your first search
doesn't find what you need, try rephrasing or looking at the full document."""


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
        model=llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
    )

    # Run the agent (track tokens/cost across all LLM calls)
    with get_openai_callback() as cb:
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": max_iterations * 2 + 1},
        )
    prompt_tokens = cb.prompt_tokens
    completion_tokens = cb.completion_tokens
    total_tokens = cb.total_tokens
    cost_usd = estimate_cost(model, prompt_tokens, completion_tokens)

    messages = result.get("messages", [])

    # Parse messages to extract answer, tool calls, and trace steps
    llm_calls = 0
    answer = "No answer produced."

    for msg in messages:
        if isinstance(msg, AIMessage):
            llm_calls += 1
            # If this AI message has tool calls, record them as steps
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    steps.append(f"THOUGHT: Agent decided to use '{tc['name']}'")
                    steps.append(f"ACTION: {tc['name']}({tc['args']})")
            else:
                # Final answer — last AIMessage without tool calls
                answer = msg.content
                steps.append(f"FINAL ANSWER: Generated response after {len(tool_calls)} tool calls")

        elif isinstance(msg, ToolMessage):
            full_output = str(msg.content)
            observation_preview = full_output[:200]
            steps.append(f"OBSERVATION: {observation_preview}...")

            # Match this observation back to the most recent AI tool call
            tool_calls.append({
                "tool": msg.name,
                "input": "",   # args captured above in AIMessage
                "output_preview": observation_preview,  # display only
                "output_full": full_output,             # full text for RAGAS context
            })

    elapsed = time.time() - start

    return AgenticResult(
        question=question,
        answer=answer,
        llm_calls=llm_calls,
        tool_calls=tool_calls,
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        latency_seconds=round(elapsed, 2),
        steps=steps,
    )
