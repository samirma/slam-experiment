from flask import Blueprint, render_template, request, jsonify
from .llm_services import generate_script_from_llm
from .script_runner import run_script_in_docker # Import the new function
import os # For os.getenv in route

main_blueprint = Blueprint('main', __name__)

@main_blueprint.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@main_blueprint.route('/generate_script', methods=['POST'])
def generate_script_route():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    user_input = data.get('user_input')
    llm_provider = data.get('llm_provider')

    if not user_input or not llm_provider:
        return jsonify({"error": "Missing user_input or llm_provider"}), 400

    generated_script = generate_script_from_llm(user_input, llm_provider)

    return jsonify({
        "generated_script": generated_script
    })

@main_blueprint.route('/run_script', methods=['POST'])
def run_script_route(): # Renamed function
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data received"}), 400
    script_to_run = data.get('script')
    if not script_to_run:
        return jsonify({"error": "No script provided"}), 400

    # Prepare environment variables to pass to the script runner
    # These are the keys the script inside Docker might need
    # The script_runner itself will also try to load them from its own env if not passed explicitly
    env_vars_to_pass = {}
    for key_name in ["OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY",
                     "TAVILY_API_KEY", "SERPER_API_KEY", "BROWSERLESS_API_KEY"]: # Common keys for CrewAI tools
        key_value = os.getenv(key_name) # Get from web app's environment (loaded from .env by main.py)
        if key_value:
            env_vars_to_pass[key_name] = key_value

    print(f"Passing API keys to script runner: {list(env_vars_to_pass.keys())}")


    stdout, stderr = run_script_in_docker(script_to_run, env_vars=env_vars_to_pass)

    response_output = ""
    has_stdout = stdout and stdout.strip()
    has_stderr = stderr and stderr.strip()

    if has_stdout:
        response_output += f"STDOUT:\n{stdout.strip()}\n"
    if has_stderr:
        response_output += f"STDERR:\n{stderr.strip()}\n"

    if not has_stdout and not has_stderr:
        response_output = "Script produced no discernible output to stdout or stderr."

    return jsonify({
        "output": response_output.strip() # Strip final potentially leading/trailing newlines from the whole message
    })
