import json

from confluent_kafka import Consumer
from langgraph.checkpoint.postgres import PostgresSaver

from resilient_agent import builder, run_task, DB_URI

consumer = Consumer({
    "bootstrap.servers": "localhost:9092",
    "group.id": "ares-agents",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,
})
consumer.subscribe(["agent-tasks"])

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
    print("Worker waiting for tasks...")

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print("Kafka error:", msg.error())
                continue

            data = json.loads(msg.value())
            result = run_task(graph, data["task_id"], data["task"])
            print(f"Finished {data['task_id']}: {result['completed_steps']}")

            consumer.commit(message=msg, asynchronous=False)
    finally:
        consumer.close()