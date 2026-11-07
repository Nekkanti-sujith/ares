import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout


class ModelError(Exception):
    pass


class AllModelsFailed(Exception):
    pass


class MockModel:
    """Fake LLM that fails on purpose, so we can test the router."""

    def __init__(self, name, failure_rate=0.0, latency=0.5):
        self.name = name
        self.failure_rate = failure_rate
        self.latency = latency

    def invoke(self, prompt: str) -> str:
        time.sleep(self.latency)
        if random.random() < self.failure_rate:
            raise ModelError(f"{self.name} failed")
        return f"[{self.name}] response to: {prompt}"


class CircuitBreaker:
    def __init__(self, max_failures=3, cooldown=30):
        self.max_failures = max_failures
        self.cooldown = cooldown
        self.failures = 0
        self.opened_at = None

    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.time() - self.opened_at >= self.cooldown:
            self.opened_at = None  # cooldown over, give it another chance
            self.failures = 0
            return False
        return True

    def record_success(self):
        self.failures = 0
        self.opened_at = None

    def record_failure(self):
        self.failures += 1
        if self.failures >= self.max_failures:
            self.opened_at = time.time()


class ModelRouter:
    def __init__(self, models, timeout=10):
        self.models = models
        self.timeout = timeout
        self.breakers = {m.name: CircuitBreaker() for m in models}
        self.executor = ThreadPoolExecutor(max_workers=4)

    def invoke(self, prompt: str) -> str:
        for model in self.models:
            breaker = self.breakers[model.name]
            if breaker.is_open():
                print(f"  {model.name}: circuit open, skipping")
                continue

            future = self.executor.submit(model.invoke, prompt)
            try:
                result = future.result(timeout=self.timeout)
                breaker.record_success()
                return result
            except FutureTimeout:
                print(f"  {model.name}: timed out")
            except Exception as e:
                print(f"  {model.name}: {e}")
            breaker.record_failure()

        raise AllModelsFailed("every model failed or is unavailable")


if __name__ == "__main__":
    router = ModelRouter(
        [
            MockModel("claude", failure_rate=0.7),
            MockModel("gpt-5", failure_rate=0.3),
            MockModel("qwen", failure_rate=0.0),
        ],
        timeout=2,
    )
    for i in range(10):
        print(f"Call {i + 1}:")
        print("  ->", router.invoke(f"question {i + 1}"))