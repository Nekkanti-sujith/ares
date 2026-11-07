import sys
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from services.router.router import ModelRouter, MockModel

DB_URI = "postgresql://ares:ares@localhost:5432/ares"


# 1. The state
class AgentState(TypedDict):
    task: str
    completed_steps: Annotated[list[str], operator.add]
    results: Annotated[list[str], operator.add]


# 2. The model router: Claude first, then GPT-5, then Qwen
router = ModelRouter(
    [
        MockModel("claude", failure_rate=0.3, latency=3),
        MockModel("gpt-5", failure_rate=0.3, latency=3),
        MockModel("qwen", failure_rate=0.0, latency=3),
    ],
    timeout=10,
)


# 3. One function that builds each step
def make_step(name: str):
    def step(state: AgentState):
        print(f"Running {name}...")
        output = router.invoke(f"{name}: {state['task']}")
        print(f"  -> {output}")
        return {"completed_steps": [name], "results": [output]}
    return step


# 4. The graph
builder = StateGraph(AgentState)
for name in ["research", "analyze", "draft", "review"]:
    builder.add_node(name, make_step(name))

builder.add_edge(START, "research")
builder.add_edge("research", "analyze")
builder.add_edge("analyze", "draft")
builder.add_edge("draft", "review")
builder.add_edge("review", END)


# 5. Run or resume a task
def run_task(graph, task_id: str, task: str):
    config = {"configurable": {"thread_id": task_id}}
    state = graph.get_state(config)
    if state.values:
        print(f"Resuming {task_id}, done so far: {state.values['completed_steps']}")
        return graph.invoke(None, config, durability="sync")
    print(f"Starting {task_id}")
    return graph.invoke(
        {"task": task, "completed_steps": [], "results": []},
        config,
        durability="sync",
    )


# 6. Manual run for testing
if __name__ == "__main__":
    task_id = sys.argv[1] if len(sys.argv) > 1 else "task-1"

    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
        graph = builder.compile(checkpointer=checkpointer)
        result = run_task(graph, task_id, "test task")

    print("Done:", result)