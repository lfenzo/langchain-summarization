from app.strategies.execution import BaseExecutionStrategy, InvokeStrategy, StreamingStrategy


class ExecutionStrategyFactory:

    def __init__(self) -> None:
        self.available_execution_strategies = {
            'stream': StreamingStrategy,
            'invoke': InvokeStrategy,
        }

    def create(self, strategy: str, **kwargs) -> BaseExecutionStrategy:
        return self.available_execution_strategies[strategy](**kwargs)
