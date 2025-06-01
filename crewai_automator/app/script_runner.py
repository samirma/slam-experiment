import subprocess
import tempfile
import os
import docker # Docker SDK for Python

# Initialize Docker client
try:
    docker_client = docker.from_env()
except docker.errors.DockerException:
    docker_client = None
    print("ERROR: Docker is not running or docker SDK is not configured correctly.")

# Define the name for the runner image (must match how it's built)
RUNNER_IMAGE_NAME = "crewai_script_runner:latest"
# Define the directory inside the runner container where scripts will be placed
CONTAINER_SCRIPT_DIR = "/script_execution"
TEMP_SCRIPT_NAME = "temp_crew_script.py"

def build_runner_image_if_not_exists():
    """Builds the runner Docker image if it doesn't exist."""
    if not docker_client:
        print("Docker client not available. Cannot build runner image.")
        return False
    try:
        docker_client.images.get(RUNNER_IMAGE_NAME)
        print(f"Runner image '{RUNNER_IMAGE_NAME}' already exists.")
        return True
    except docker.errors.ImageNotFound:
        print(f"Runner image '{RUNNER_IMAGE_NAME}' not found. Attempting to build...")
        try:
            # Assuming Dockerfile.runner is in the parent directory of this script's directory (project root)
            # dockerfile_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Dockerfile.runner') # Not directly used by build, path is for context
            project_root = os.path.dirname(os.path.dirname(__file__)) # crewai_automator directory

            # The context for the Docker build should be the project root,
            # so it can access app/runner_requirements.txt
            print(f"Building Docker image '{RUNNER_IMAGE_NAME}' with context '{project_root}' and Dockerfile 'Dockerfile.runner'")
            response = docker_client.images.build(
                path=project_root,
                dockerfile='Dockerfile.runner', # Relative to the context path
                tag=RUNNER_IMAGE_NAME,
                rm=True # Remove intermediate containers
            )
            print(f"Runner image '{RUNNER_IMAGE_NAME}' built successfully.")
            # for line in response[1]: # Print build logs if needed
            #    if 'stream' in line:
            #        print(line['stream'].strip())
            return True
        except docker.errors.BuildError as e:
            print(f"Error building runner image '{RUNNER_IMAGE_NAME}': {e}")
            for line in e.build_log:
                if 'stream' in line: # Build log is a list of dicts
                    print(line['stream'].strip())
            return False
        except Exception as e:
            print(f"An unexpected error occurred during image build: {e}")
            return False
    except Exception as e:
        print(f"Error checking for runner image: {e}")
        return False

def run_script_in_docker(python_script_content: str, env_vars: dict = None):
    """
    Runs the given Python script content in an isolated Docker container.
    `env_vars` is a dictionary of environment variables to set in the container.
    """
    if not docker_client:
        return "Error: Docker client not initialized. Cannot run script.", ""

    if not build_runner_image_if_not_exists():
        return "Error: Runner image could not be built or found. Cannot run script.", ""

    # Create a temporary file to hold the script content on the host
    # It's created in the current working directory of the Flask app (crewai_automator/app)
    # The absolute path is then used for the Docker volume mount.
    host_script_path = ""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=".") as tmp_script_file:
            tmp_script_file.write(python_script_content)
            host_script_path = os.path.abspath(tmp_script_file.name)
            tmp_script_file.flush()
            os.fsync(tmp_script_file.fileno())

        container_script_path = f"{CONTAINER_SCRIPT_DIR}/{TEMP_SCRIPT_NAME}"

        environment = env_vars if env_vars else {}
        # Propagate essential API keys if not already in env_vars from routes.py
        # This is a fallback, routes.py should primarily handle passing these.
        for key_name in ["OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY",
                         "TAVILY_API_KEY", "SERPER_API_KEY", "BROWSERLESS_API_KEY"]:
            if key_name not in environment: # Only if not already set by the caller
                key_value = os.getenv(key_name)
                if key_value:
                    environment[key_name] = key_value

        print(f"Attempting to run script in Docker. Host path: {host_script_path}, Container path: {container_script_path}")
        print(f"Using image: {RUNNER_IMAGE_NAME}")
        # print(f"Environment variables for container: {list(environment.keys())}") # Don't log values for security

        container_logs = docker_client.containers.run(
            RUNNER_IMAGE_NAME,
            command=["python", TEMP_SCRIPT_NAME],
            volumes={
                host_script_path: {
                    "bind": container_script_path,
                    "mode": "ro"
                }
            },
            working_dir=CONTAINER_SCRIPT_DIR,
            environment=environment,
            remove=True,
            stdout=True,
            stderr=True,
            user="scriptuser"
        )
        # The 'container_logs' here will be the combined stdout and stderr if the container runs successfully.
        # For more distinct streams on error, ContainerError is caught.
        stdout = container_logs.decode('utf-8')
        stderr = "" # If successful, stderr is usually part of stdout or empty.
        print(f"Script executed. Raw output from container:\n{stdout}")
        return stdout, stderr

    except docker.errors.ContainerError as e:
        error_message = f"Error running script in Docker container (exit code {e.exit_status}):\n"
        stdout_decoded = e.stdout.decode('utf-8') if e.stdout else ""
        stderr_decoded = e.stderr.decode('utf-8') if e.stderr else "Container exited with non-zero status or no specific stderr."
        error_message += f"STDOUT:\n{stdout_decoded}\n"
        error_message += f"STDERR:\n{stderr_decoded}"
        print(error_message)
        return stdout_decoded, stderr_decoded
    except docker.errors.ImageNotFound:
        print(f"Error: Runner image '{RUNNER_IMAGE_NAME}' not found during run.")
        return "Error: Docker runner image not found. Please build it first.", ""
    except Exception as e:
        error_message = f"An unexpected error occurred while running script in Docker: {str(e)}"
        print(error_message)
        return f"Unexpected error: {str(e)}", ""
    finally:
        if os.path.exists(host_script_path):
            try:
                os.remove(host_script_path)
                print(f"Temporary script file {host_script_path} removed.")
            except OSError as e:
                print(f"Error removing temporary script file {host_script_path}: {e}")

if __name__ == '__main__':
    # For direct testing of this script_runner.py
    # Ensure:
    # 1. Docker is running.
    # 2. `pip install docker python-dotenv` in your environment.
    # 3. You are in the `crewai_automator` directory.
    # 4. Run as a module: `python -m app.script_runner`
    # 5. A `.env` file in `crewai_automator` with API keys can be used for testing key propagation.

    print("--- Direct Test of script_runner.py ---")
    # Load .env from project root for this test script to access API keys
    from dotenv import load_dotenv
    project_root_env = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(project_root_env):
       load_dotenv(dotenv_path=project_root_env)
       print(f".env loaded for script_runner direct test. OPENAI_API_KEY is set: {os.getenv('OPENAI_API_KEY') is not None}")
    else:
       print(f"Test .env file not found at {project_root_env}, API keys might not be passed to container.")

    print("Attempting to build/ensure runner image exists...")
    if build_runner_image_if_not_exists():
        print("\nTesting script execution in Docker...")
        test_script_content = """
import os
print("Hello from inside the Docker container!")
print(f"Current working directory: {os.getcwd()}")
print(f"User: {os.getenv('USER', 'N/A')}") # USER env var might not be set, use whoami or id
print("Files in current directory:")
print(os.listdir("."))
print(f"--- Environment Variables ---")
# print(os.environ) # Be careful printing all env vars
print(f"TEST_VAR_FROM_RUNNER is set: {os.getenv('TEST_VAR_FROM_RUNNER') is not None}")
print(f"OPENAI_API_KEY is set: {os.getenv('OPENAI_API_KEY') is not None}")
print(f"GEMINI_API_KEY is set: {os.getenv('GEMINI_API_KEY') is not None}")
print("--- End Environment Variables ---")

# Test importing a library from runner_requirements.txt
try:
    from dotenv import load_dotenv
    print("Successfully imported dotenv from container.")
except ImportError:
    print("Failed to import dotenv from container.")

# Example of a simple crewai script (requires API keys to be set in environment)
# Note: Actual execution of a full crew might be slow for a simple test.
# This part primarily tests if crewai can be imported and basic setup works.
print("Attempting to import crewai...")
try:
    from crewai import Agent, Task, Crew
    print("Successfully imported Agent, Task, Crew from crewai.")
    # if os.getenv('OPENAI_API_KEY'):
    #     print("OPENAI_API_KEY is set, attempting to initialize Agent (minimal test)...")
    #     try:
    #         researcher = Agent(
    #             role='Researcher',
    #             goal='Find interesting facts',
    #             backstory='You are a world class researcher',
    #             verbose=False # Keep it quiet for this test
    #         )
    #         print("CrewAI Agent initialized (minimal).")
    #     except Exception as e:
    #         print(f"Error initializing CrewAI Agent: {e}")
    # else:
    #    print("Skipping CrewAI Agent initialization as OPENAI_API_KEY is not set in container.")
except ImportError as e:
    print(f"Failed to import from crewai: {e}")
except Exception as e:
    print(f"An unexpected error related to crewai import: {e}")

print("Script finished.")
"""
        test_env_vars = {"TEST_VAR_FROM_RUNNER": "HelloContainerFromTest"}
        # Propagate API keys loaded by the __main__ block's load_dotenv
        if os.getenv("OPENAI_API_KEY"): test_env_vars["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        if os.getenv("GEMINI_API_KEY"): test_env_vars["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

        stdout, stderr = run_script_in_docker(test_script_content, env_vars=test_env_vars)
        print("\n--- Docker Script Execution STDOUT ---")
        print(stdout)
        print("--- Docker Script Execution STDERR ---")
        print(stderr if stderr else "[No stderr output]")
    else:
        print("Could not build or find runner image. Skipping execution test.")
    print("--- End of Direct Test ---")
