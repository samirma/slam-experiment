# CrewAI Automator

CrewAI Automator is a web application designed to streamline the process of generating and running CrewAI scripts. Users can provide initial instructions through a web interface, select an LLM (like Gemini or ChatGPT), and the application will generate the corresponding CrewAI Python script. This generated script can then be executed in an isolated Docker environment directly from the web interface.

## Features

*   Web interface for providing instructions and selecting LLMs.
*   Supports multiple LLMs for script generation (currently Gemini, ChatGPT; DeepSeek placeholder).
*   Generates CrewAI Python scripts based on user input and a configurable reference/template (`crewai_reference.md`).
*   Executes generated scripts in an isolated Docker container.
*   Displays script output and execution logs in the web interface.
*   API keys are managed via a `.env` file.

## Project Structure

```
crewai_automator/
├── app/                    # Main Flask application code
│   ├── __init__.py
│   ├── main.py             # Flask app entry point
│   ├── routes.py           # Web routes
│   ├── llm_services.py     # LLM interaction logic
│   ├── script_runner.py    # Logic for running scripts in Docker
│   ├── templates/          # HTML templates
│   │   └── index.html
│   ├── static/             # Static files (CSS, JS - currently minimal)
│   └── runner_requirements.txt # Python requirements for the script execution container
├── Dockerfile              # Dockerfile for the main web application
├── Dockerfile.runner       # Dockerfile for the script execution environment
├── requirements.txt        # Python requirements for the web application
├── .env_sample             # Sample environment file
├── .env                    # Environment file for API keys (gitignored)
├── crewai_reference.md     # IMPORTANT: User-defined meta-prompt/template for LLM script generation
└── README.md               # This file
```

## Setup and Configuration

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd crewai_automator
    ```

2.  **Create Environment File:**
    Copy the sample `.env_sample` file to `.env` and fill in your API keys:
    ```bash
    cp .env_sample .env
    ```
    Edit `.env` with your actual keys:
    ```
    OPENAI_API_KEY=your_openai_api_key
    GEMINI_API_KEY=your_gemini_api_key
    DEEPSEEK_API_KEY=your_deepseek_api_key # Optional, as DeepSeek is not fully implemented

    # Add any other API keys your CrewAI scripts might need for tools, e.g.:
    # TAVILY_API_KEY=your_tavily_api_key
    # SERPER_API_KEY=your_serper_api_key
    # BROWSERLESS_API_KEY=your_browserless_api_key
    ```

3.  **Update `crewai_reference.md`:**
    This file is crucial for guiding the LLM in generating scripts. The placeholder content should be **replaced or modified by you** to define how the 'Initial Instruction Input' from the web interface is transformed into a runnable CrewAI script.

    *   **If your 'Initial Instruction Input' IS the complete meta-prompt for the LLM:**
        You might modify `app/llm_services.py` to directly use the user's input as the main prompt, potentially ignoring `crewai_reference.md` or using it for common boilerplate only. (This was based on recent user feedback and is the recommended interpretation).
    *   **If `crewai_reference.md` is a static detailed meta-prompt:**
        The current code in `app/llm_services.py` reads this file and combines it with the 'Initial Instruction Input'. Ensure this file contains your detailed instructions for the LLM.

    The placeholder in `crewai_reference.md` provides guidance on what this file should ideally contain.

## Running the Application

You can run the application either directly using Flask (for development) or using Docker.

### Option 1: Running without Docker (Local Development)

1.  **Create a Python Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scriptsctivate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This installs Flask, Docker SDK, LLM libraries, etc.

3.  **Ensure Docker is Running (for Script Execution):**
    Even if you run the web app locally, the script execution part (`script_runner.py`) uses Docker to run the generated CrewAI scripts. Make sure your Docker daemon is running and accessible.

4.  **Run the Flask Application:**
    ```bash
    python -m app.main
    ```
    The application should be accessible at `http://0.0.0.0:5000` or `http://localhost:5000`.

### Option 2: Running with Docker

This method containerizes the web application and its script execution capabilities.

1.  **Ensure Docker is Running.**

2.  **Build the Docker Images:**
    *   The main application's Docker image (`crewai_automator:latest` defined in `Dockerfile`) will be built first.
    *   The script runner image (`crewai_script_runner:latest` defined in `Dockerfile.runner`) is designed to be built automatically by the application on its first attempt to run a script if it doesn't already exist. However, you can also pre-build it if you modify `script_runner.py` or want to ensure it's ready. For now, rely on the automatic build by `script_runner.py`.

    To build the main application image (optional, as `docker-compose up --build` would also do this if using Compose):
    ```bash
    docker build -t crewai_automator:latest .
    ```

3.  **Run the Main Application Container:**
    You need to pass your API keys as environment variables to the container and mount the Docker socket to allow the application to manage other Docker containers (for script execution).

    ```bash
    docker run -d \
        -p 5000:5000 \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -e OPENAI_API_KEY="your_openai_api_key" \
        -e GEMINI_API_KEY="your_gemini_api_key" \
        # Add other -e flags for any other API keys needed by your CrewAI scripts
        --name crewai_automator_app \
        crewai_automator:latest
    ```
    *   `-d`: Run in detached mode.
    *   `-p 5000:5000`: Map port 5000 from the container to port 5000 on your host.
    *   `-v /var/run/docker.sock:/var/run/docker.sock`: Mount the Docker socket.
    *   `-e API_KEY_NAME="value"`: Pass environment variables.
    *   `--name crewai_automator_app`: Assign a name to the container.

    The application should then be accessible at `http://localhost:5000`.

4.  **Script Execution Container (`Dockerfile.runner`):**
    As mentioned, `script_runner.py` (running inside the `crewai_automator_app` container) will attempt to build the `crewai_script_runner:latest` image using `/app/Dockerfile.runner` (which is inside the main app container) if it doesn't find it. This build uses requirements from `/app/app/runner_requirements.txt`. Generated scripts are then run in new containers created from this `crewai_script_runner:latest` image.

## Development Notes

*   **DeepSeek LLM:** Integration for DeepSeek is currently a placeholder in `app/llm_services.py` and needs to be fully implemented if required.
*   **Error Handling:** Basic error handling is in place. Further enhancements can be added for more robustness.
*   **Frontend:** The HTML interface is basic. Enhancements like loading indicators, better styling, or using a JavaScript framework could improve user experience.
*   **Security:** Running arbitrary code always carries risks. The current setup uses Docker containers for isolation. Ensure your Docker environment is secure. API keys should be handled carefully and not exposed.

## Contributing
(Placeholder for contribution guidelines if this were a public project)

```
