import os
import google.generativeai as genai
from openai import OpenAI

# Environment variables are expected to be loaded by main.py
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") # Still here, but not implemented yet

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Configure OpenAI (for ChatGPT)
# DeepSeek might also use this client if its API is OpenAI-compatible
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

deepseek_client = None
if DEEPSEEK_API_KEY:
    # Placeholder: If DeepSeek has its own client or uses OpenAI's
    # For now, assuming it might use a similar structure or require a custom one
    # If DeepSeek is OpenAI compatible, the existing openai_client could be used,
    # but it would need a different base_url.
    # For now, let's assume we might need a separate client or a modified one.
    # deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="DEEPSEEK_API_BASE_URL")
    pass


def get_crewai_reference_content():
    """Reads the content of crewai_reference.md."""
    try:
        # Assuming crewai_reference.md is in the project root (crewai_automator)
        # and this script is in crewai_automator/app/
        ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'crewai_reference.md')
        with open(ref_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "# Error: crewai_reference.md not found."
    except Exception as e:
        return f"# Error reading crewai_reference.md: {str(e)}"

def generate_script_from_llm(user_input: str, llm_provider: str):
    crew_reference_content = get_crewai_reference_content()
    if crew_reference_content.startswith("# Error:"):
        return crew_reference_content

    prompt = f"{crew_reference_content}\n\nUser Instructions:\n{user_input}\n\nGenerate Python script based on the above reference and instructions for CrewAI."

    if llm_provider == "gemini":
        if not GEMINI_API_KEY:
            return "# Error: GEMINI_API_KEY not configured."
        try:
            model = genai.GenerativeModel('gemini-pro') # Or other appropriate model
            response = model.generate_content(prompt)
            # Basic check for safety ratings if applicable and response structure
            if response.candidates and response.candidates[0].content.parts:
                 generated_text = "".join(part.text for part in response.candidates[0].content.parts)
                 # Remove markdown code block formatting if present
                 if generated_text.startswith("```python"):
                     generated_text = generated_text[len("```python"):].strip()
                 if generated_text.endswith("```"):
                     generated_text = generated_text[:-len("```")].strip()
                 return generated_text
            else:
                # Fallback or error if no valid content
                # Check for prompt feedback which might indicate blocking
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    return f"# Error: Gemini generation blocked due to: {response.prompt_feedback.block_reason}"
                return "# Error: Gemini generated no valid content."
        except Exception as e:
            return f"# Error during Gemini API call: {str(e)}"

    elif llm_provider == "chatgpt":
        if not openai_client:
            return "# Error: OPENAI_API_KEY not configured or client not initialized."
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo", # Or "gpt-4" etc.
                messages=[
                    {"role": "system", "content": "You are an assistant that generates Python scripts for CrewAI based on a reference meta-prompt and user instructions."},
                    {"role": "user", "content": prompt}
                ]
            )
            generated_text = response.choices[0].message.content.strip()
            # Remove markdown code block formatting if present
            if generated_text.startswith("```python"):
                generated_text = generated_text[len("```python"):].strip()
            if generated_text.endswith("```"):
                generated_text = generated_text[:-len("```")].strip()
            return generated_text
        except Exception as e:
            return f"# Error during OpenAI API call: {str(e)}"

    elif llm_provider == "deepseek":
        if not DEEPSEEK_API_KEY: # Basic check
             return "# Error: DEEPSEEK_API_KEY not configured."
        # TODO: Implement DeepSeek API call
        # This might involve using the OpenAI client with a custom base_url,
        # or a different library/method if DeepSeek's API differs significantly.
        # Example if using OpenAI client with custom base_url:
        # try:
        #     if not deepseek_client: # This client would need to be initialized with base_url
        #          # deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="YOUR_DEEPSEEK_API_ENDPOINT")
        #          return "# Error: DeepSeek client not properly configured with base URL."
        #     response = deepseek_client.chat.completions.create(
        #         model="deepseek-coder", # Or other appropriate DeepSeek model
        #         messages=[
        #             {"role": "system", "content": "You are an assistant that generates Python scripts for CrewAI based on a reference meta-prompt and user instructions."},
        #             {"role": "user", "content": prompt}
        #         ]
        #     )
        #     return response.choices[0].message.content.strip()
        # except Exception as e:
        #     return f"# Error during DeepSeek API call: {str(e)}"
        return "# DeepSeek integration is not yet fully implemented."
    else:
        return "# Error: Unknown LLM provider selected."

# For direct testing:
if __name__ == '__main__':
    # Ensure .env is in ../ (crewai_automator directory) relative to this file (crewai_automator/app/llm_services.py)
    # Example: crewai_automator/.env should have GEMINI_API_KEY or OPENAI_API_KEY
    # from dotenv import load_dotenv
    # load_dotenv(dotenv_path='../.env') # Load .env from parent for direct script run

    # print("Testing LLM Services Directly:")
    # print(f"Gemini Key Loaded: {os.getenv('GEMINI_API_KEY') is not None}")
    # print(f"OpenAI Key Loaded: {os.getenv('OPENAI_API_KEY') is not None}")

    # print("\n--- CrewAI Reference Content ---")
    # print(get_crewai_reference_content())

    # print("\n--- Testing Gemini ---")
    # if os.getenv("GEMINI_API_KEY"):
    #     print(generate_script_from_llm("Create a simple task for a research agent.", "gemini"))
    # else:
    #     print("Skipping Gemini test: GEMINI_API_KEY not found in .env")

    # print("\n--- Testing ChatGPT (OpenAI) ---")
    # if os.getenv("OPENAI_API_KEY"):
    #     print(generate_script_from_llm("Create a simple task for a writer agent.", "chatgpt"))
    # else:
    #     print("Skipping ChatGPT test: OPENAI_API_KEY not found in .env")

    # print("\n--- Testing DeepSeek (Placeholder) ---")
    # print(generate_script_from_llm("Test DeepSeek.", "deepseek"))
    pass
