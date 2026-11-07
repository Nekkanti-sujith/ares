import time
import operator
from typing import Annotated, TypedDict
import sys
from langgraph.checkpoint.postgres import PostgresSaver

from langgraph.graph import StateGraph, START, END

DB_URI = "postgresql://ares:ares@localhost:5432/ares"

# 1. The state: data passed between step
class AgentState(TypedDict):
    task: str
    completed_steps: Annotated[list[str], operator.add]


# 2. The steps: each one is a plain function
def research(state: AgentState):
    print("Running research...")
    time.sleep(5)  # pretend this is an LLM call
    return {"completed_steps": ["research"]}

def analyze(state: AgentState):
    print("Running analyze...")
    time.sleep(5)
    return {"completed_steps": ["analyze"]}

def draft(state: AgentState):
    print("Running draft...")
    time.sleep(5)
    return {"completed_steps": ["draft"]}

def review(state: AgentState):
    print("Running review...")
    time.sleep(5)
    return {"completed_steps": ["review"]}


# 3. The graph: connect the steps in order
builder = StateGraph(AgentState)
builder.add_node("research", research)
builder.add_node("analyze", analyze)
builder.add_node("draft", draft)
builder.add_node("review", review)

builder.add_edge(START, "research")
builder.add_edge("research", "analyze")
builder.add_edge("analyze", "draft")
builder.add_edge("draft", "review")
builder.add_edge("review", END)

# 4. Run it
if __name__ == "__main__":
    
    task_id = sys.argv[1] if len(sys.argv) > 1 else "task-1"
    config = {"configurable": {"thread_id": task_id}}

    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()  # creates checkpoint tables (safe to run every time)
        graph = builder.compile(checkpointer=checkpointer)

        state = graph.get_state(config)
        if state.values:
            print(f"Resuming {task_id}, done so far: {state.values['completed_steps']}")
            result = graph.invoke(None, config, durability="sync")
        else:
            print(f"Starting {task_id}")
            result = graph.invoke(
                {"task": "test task", "completed_steps": []},
                config,
                durability="sync",
            )
    print("Done:", result)