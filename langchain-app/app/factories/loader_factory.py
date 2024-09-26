from langchain_core.document_loaders import BaseLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import FasterWhisperParser


class LoaderFactory:
    """
    Factory class for creating document loader instances based on the file type (MIME type).

    Attributes
    ----------
    loader_from_mime_type : dict
        A dictionary mapping MIME types (str) to their respective loader factory methods.
    """

    def __init__(self):
        self.loader_from_mime_type = {
            'application/pdf': self._get_pdf_loader,
            'video/mp4': self._get_audio_loader,
        }

    def create(self, file_type: str, file_path: str, **kwargs) -> BaseLoader:
        """
        Create a document loader instance based on the specified file type (MIME type).

        Parameters
        ----------
        file_type : str
            The MIME type of the file (e.g., 'application/pdf').
        file_path : str
            The path to the file that needs to be loaded.
        **kwargs : dict
            Additional keyword arguments passed to the loader class.

        Returns
        -------
        BaseLoader
            The document loader instance created.

        Raises
        ------
        ValueError
            If the specified file type is not valid.

        Examples
        --------
        >>> factory = LoaderFactory()
        >>> loader = factory.create('application/pdf', '/path/to/file.pdf')
        """
        if file_type not in self.loader_from_mime_type:
            raise ValueError(
                f"Invalid file type '{file_type}'. "
                f"Valid file types are: {self.get_valid_mime_types()}"
            )
        return self.loader_from_mime_type[file_type](file_path=file_path, **kwargs)

    def _get_pdf_loader(self, file_path: str, **kwargs) -> PyMuPDFLoader:
        """
        Creates a PyMuPDFLoader instance for loading PDF documents.

        Parameters
        ----------
        file_path : str
            The path to the PDF file.
        **kwargs : dict
            Additional keyword arguments for configuring the loader.

        Returns
        -------
        PyMuPDFLoader
            The loader instance for handling PDF documents.
        """
        return PyMuPDFLoader(file_path=file_path, **kwargs)

    def _get_audio_loader(self, file_path: str, model_size: str = 'large-v3') -> GenericLoader:
        """
        Creates a GenericLoader instance for loading audio files.

        The audio files are processed using the FasterWhisperParser.

        Parameters
        ----------
        file_path : str
            The path to the audio file.
        model_size : str, optional
            The model size for the FasterWhisperParser (default is 'large-v3').

        Returns
        -------
        GenericLoader
            The loader instance for handling audio files.
        """
        return GenericLoader.from_filesystem(
            path=file_path,
            parser=FasterWhisperParser(model_size=model_size),
        )

    def get_valid_mime_types(self) -> list[str]:
        """
        Get a list of valid MIME types that can be used to create loaders.

        Returns
        -------
        list[str]
            A list of valid MIME type keys.
        """
        return list(self.loader_from_mime_type.keys())

