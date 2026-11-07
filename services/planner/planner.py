import json
import sys
import uuid

from confluent_kafka import Producer

producer = Producer({"bootstrap.servers": "localhost:9092"})


def publish_task(task: str) -> str:
    task_id = f"task-{uuid.uuid4().hex[:8]}"
    message = {"task_id": task_id, "task": task}
    producer.produce("agent-tasks", key=task_id, value=json.dumps(message))
    producer.flush()  # wait until Kafka confirms it received the message
    return task_id


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "test task"
    print("Published", publish_task(task))