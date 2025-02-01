# Langchain Summarization App

This repository provides an example of a modular, containerized system for performing summarization
tasks on text, audio, and video content using [LangChain](https://github.com/langchain-ai/langchain).
The application is designed to process documents through a microservice architecture managed with
Podman Compose, leveraging GPU-accelerated language models and supporting caching and storage.

## Features

- **Multi-format support:** Processes text, audio, and video files with dynamic Loader selection based on file type.
- **Streaming & Batching:** Supports both streaming summaries and batch processing based on user preferences.
- **GPU-accelerated services:** Separates heavy GPU dependencies into their own containers for efficient resource utilization.
- **Modular architecture:** Enables easy swapping or scaling of services, such as replacing the LLM or transcription backends.

For more detailed information read [this article](https://lfenzo.github.io/projects/deep-learning/langchain-summarization-app/).

## Requirements

- **Podman:** Install [Podman](https://podman.io/) and [Podman Compose](https://github.com/containers/podman-compose).
- **NVIDIA Container Toolkit:** Required for GPU-enabled containers. Refer to the [NVIDIA documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Setup and Deployment

1) Clone the repository
    ```bash
    git clone https://github.com/lfenzo/langchain-summarization.git
    cd langchain-summarization
    ```

1) Start the containers with Podman Compose
    ```bash
    podman compose up 
    ```
    Ensure that you system supports GPU passthough for the LLM and Transcription services.

1) Make sure that the LLM of your choice is pulled in the Ollama Server:
    ```bash
    podman exec langchain-summarization_ollama-server_1 ollama pull <your_llm_here>
    ```

1) Test the summarization endpoint in the FastAPI service available at `http://0.0.0.0:8000`:
    ```bash
    curl -X POST "http://0.0.0.0:8000/summarize" \
         -F "file=@path/to/your/file.pdf;type=application/octet-stream" \
         --no-buffer
    ```
