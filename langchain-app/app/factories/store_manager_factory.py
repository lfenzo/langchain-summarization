from app.storage.mongodb import MongoDBStoreManager


class StoreManagerFactory:
    """
    Factory class for creating store manager instances.

    Attributes
    ----------
    store_managers : dict
        A dictionary mapping store manager names (str) to their respective classes.
    """

    def __init__(self):
        self.store_managers = {
            'mongodb': MongoDBStoreManager,
        }

    def create(self, store_manager: str, **kwargs):
        """
        Create a store manager instance based on the specified type.

        Parameters
        ----------
        store_manager : str
            The store manager type to create (e.g., 'mongodb').
        **kwargs : dict
            Additional keyword arguments passed to the store manager class.

        Returns
        -------
        Any
            The store manager instance created.

        Raises
        ------
        ValueError
            If the specified store manager type is not valid.

        Examples
        --------
        >>> factory = StoreManagerFactory()
        >>> store_manager = factory.create('mongodb', user='admin', password='admin123')
        """
        if store_manager not in self.store_managers:
            raise ValueError(
                f"Invalid store manager '{store_manager}'. "
                f"Valid store managers are: {self.get_valid_store_managers()}"
            )
        return self.store_managers[store_manager](**kwargs)

    def get_valid_store_managers(self) -> list[str]:
        """
        Get a list of valid store managers that can be created.

        Returns
        -------
        list[str]
            A list of valid store manager keys.
        """
        return list(self.store_managers.keys())
