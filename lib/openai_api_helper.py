import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any, Optional

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
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('BASE_URL')
    local_llm = os.getenv('CLOUD_LLM')

    # Validate required configuration
    if not base_url:
        print("Error: No base URL found - please add a BASE_URL to your .env file!")
        return None

    if not local_llm:
        print("Error: No local LLM model specified - please add a LOCAL_LLM to your .env file!")
        return None

    # Create messages for the LLM request
    messages = create_request_for_llm(system_prompt, user_prompt)

    # Check if api_key is None and handle it appropriately
    if api_key is None:
        print("Error: API key is required - please add an LMSTUDIO_OPENAI_API_KEY to your .env file!")
        return None

    # Call the OpenAI API
    try:
        response = call_openai_api(local_llm, base_url, api_key, messages)
        return response
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


def create_request_for_llm(system_prompt, user_prompt):
    """
    Create the messages list for the LLM request.

    Args:
        system_prompt: The system prompt for the LLM
        user_prompt: The user prompt for the LLM

    Returns:
        A list of message dictionaries for the OpenAI API
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_openai_api(model, base_url, api_key, messages):
    """
    Call the OpenAI API with the given parameters.

    Args:
        model: The model name to use
        base_url: The base URL for the API
        api_key: The API key for authentication
        messages: The messages to send to the LLM

    Returns:
        The content of the LLM's response

    Raises:
        Exception: If the API call fails
    """
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(model=model, messages=messages)
    print(response.choices[0].message.content)