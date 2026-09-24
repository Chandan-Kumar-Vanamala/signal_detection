"""Check that the OpenAI key and model in .env work, with one tiny request."""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
model = os.environ["OPENAI_MODEL"]

client = OpenAI()   # reads OPENAI_API_KEY from the environment by itself

response = client.responses.create(
    model=model,
    input="Reply with exactly: connection works",
)

print(f"Model : {response.model}")
print(f"Reply : {response.output_text}")
print(f"Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
