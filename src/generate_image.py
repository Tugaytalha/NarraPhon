import time
import requests
from typing import Optional
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API token from environment variables
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

def api_sd_generate(
    prompt: str,
    width: Optional[int] = 1024,
    height: Optional[int] = 1024,
    negative_prompt: Optional[str] = None
) -> Image.Image:
    """
    Generate an image using FLUX.1-dev via Hugging Face's inference API.
    
    Args:
        prompt (str): The text prompt for image generation
        
    Returns:
        PIL.Image: Generated image
    """
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3-medium-diffusers"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    print("Hugging Face API token loaded")
    
    # Prepare the payload (simplified for FLUX.1-dev)
    payload = {
        "inputs": prompt
    }

    try:
        # Make the API request
        response = requests.post(API_URL, headers=headers, json=payload)
        time.sleep(7)
        print("Request made to Hugging Face API")
        
        # Print debug information if the request fails
        if response.status_code != 200:
            print(f"Error Status Code: {response.status_code}")
            print(f"Response Content: {response.text}")
        
        response.raise_for_status()  # Raise an exception for bad status codes

        print("Response received from Hugging Face API")
        
        # Convert the response to an image
        image = Image.open(BytesIO(response.content))
        
        return image
    
    except requests.exceptions.RequestException as e:
        print(f"Error making request to Hugging Face API: {e}")
        raise
    except Exception as e:
        print(f"Error processing image: {e}")
        raise