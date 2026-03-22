import os
from dotenv import load_dotenv
from lib.openai_api_helper import load_and_run
from lib.openai_helper_local import load_and_run
from utilities.sanitizers import sanitize_email_text

def main():
    system_prompt = """
                    Act as a professional email assistant. Revise the following email to be more professional, concise, and impactful while preserving the core message. Respond with the email only, don't include any extra content in the response.
                    """
    user_prompt = sanitize_email_text(input("Enter your email: "))

    # Load environment variables from .env file
    load_dotenv(override=True)

    # Get configuration from environment
    inference_type = os.getenv('INFERENCE_PROVIDER')

    if inference_type == "cloud":
        load_and_run(system_prompt=system_prompt, user_prompt=user_prompt)
    elif inference_type == "local":
        load_and_run(system_prompt=system_prompt, user_prompt=user_prompt)
    else:
        print("Invalid inference provider, please check your environment file")


if __name__ == "__main__":
    main()
