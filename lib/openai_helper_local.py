import os
from dotenv import load_dotenv
import lmstudio as lms

def load_and_run(system_prompt, user_prompt):
    """
    Load environment variables and run the OpenAI API request.

    Args:
        system_prompt: The system prompt for the LLM
        user_prompt: The user prompt for the LLM

    Returns:
        The response content from the LLM, or None if an error occurred
    """
    # Load environment variables from .env file
    load_dotenv(override=True)

    # Get configuration from environment
    # api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('LMSTUDIO_BASE_URL')
    local_llm = os.getenv('LMSTUDIO_LOCAL_LLM')
    model = lms.llm(local_llm)

    # Validate required configuration
    if not base_url:
        print("Error: No base URL found - please add a BASE_URL to your .env file!")
        return None

    if not local_llm:
        print("Error: No local LLM model specified - please add a LOCAL_LLM to your .env file!")
        return None

    # Call the lmstudio API
    try:
        chat = lms.Chat(system_prompt)
        chat.add_user_message(user_prompt)

        response = model.respond(chat)
        print(response)
    except Exception as e:
        print(f"Error calling LM Studio API: {e}")
        return None
