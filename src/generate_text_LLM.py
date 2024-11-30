import os
from groq import Groq
from dotenv import load_dotenv
from typing import Dict, Optional, Any

print(f"Loading environment variables...")
load_dotenv()
print(f"Environment GROQ_API_KEY source: {os.environ.get('GROQ_API_KEY', 'Not found')[:6]}...")

# Get the Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class LLMClient:
    def __init__(self, model_version: str = "llama-3.1-8b-instant"):
        """
        Initialize the Groq client for Llama 3.1 model.

        Args:
            model_version (str): The specific Llama model version to use.
        """
        self.client = Groq()
        print("Groq client initialized")
        self.model_version = model_version

    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 2048,
        temperature: float = 0.87,
        top_p: float = 0.47
    ) -> str:
        """
        Generate text using the Llama 3.1 model.

        Args:
            prompt (str): The input prompt.
            max_tokens (int): Maximum number of tokens in the response.
            temperature (float): Sampling temperature for creativity.
            top_p (float): Nucleus sampling probability threshold.

        Returns:
            str: Generated text.
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_version,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=False
            )
            print(f"Generated text: {completion.choices[0].message.content}")
            
            return completion.choices[0].message.content
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error occurred while generating response"