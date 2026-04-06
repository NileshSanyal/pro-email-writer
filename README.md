## Local Development Envionment Setup Guide
Prequisites
- Local LLM downloaded in LM Studio (preferably Llama 3.2 1B)
- Run the LLM from LM Studio
- Expose the LLM as local inference from LM Studio by running the server

## Running Project locally

Prequisites
- Install uv (Visit: https://docs.astral.sh/uv/getting-started/installation/)
- Make sure to check if uv is installed
- Run "uv sync" command from root directory of this project
- Create a .env file at root directory of this project
- Run the command "uv run main.py" 
- Paste sample email content present in test-emails folder and press Enter button, that's it!


## Sample contents of .env file
INFERENCE_PROVIDER=cloud
#cloud settings
OPENAI_API_KEY=test-key
BASE_URL=https://sample-url
CLOUD_LLM=cloud-llm

#local settings
LMSTUDIO_BASE_URL=http://localhost-url
LMSTUDIO_LOCAL_LLM=local-llm