from app.strategies.execution import BaseExecutionStrategy, InvokeStrategy, StreamingStrategy


class ExecutionStrategyFactory:
    """
    Factory class for creating execution strategy instances.

    Attributes
    ----------
    available_execution_strategies : dict
        A dictionary mapping execution strategy names (str) to their respective classes.
    """

    def __init__(self) -> None:
        self.available_execution_strategies = {
            'stream': StreamingStrategy,
            'invoke': InvokeStrategy,
        }

    def create(self, strategy: str, **kwargs) -> BaseExecutionStrategy:
        """
        Create an execution strategy instance based on the specified strategy type.

        Parameters
        ----------
        strategy : str
            The execution strategy type to create (e.g., 'stream').
        **kwargs : dict
            Additional keyword arguments passed to the execution strategy class.

        Returns
        -------
        BaseExecutionStrategy
            The execution strategy instance created.

        Raises
        ------
        ValueError
            If the specified execution strategy type is not valid.

        Examples
        --------
        >>> factory = ExecutionStrategyFactory()
        >>> execution_strategy = factory.create('stream')
        """
        if strategy not in self.available_execution_strategies:
            raise ValueError(
                f"Invalid execution strategy '{strategy}'. "
                f"Valid execution strategies are: {self.get_valid_execution_strategies()}"
            )
        return self.available_execution_strategies[strategy](**kwargs)

    def get_valid_execution_strategies(self) -> list[str]:
        """
        Get a list of valid execution strategies that can be created.

        Returns
        -------
        list[str]
            A list of valid execution strategy keys.
        """
        return list(self.available_execution_strategies.keys())

