import time
from src.constructor import generate_presentation 
from src.prompt_configs import en_gigachat_config
from src.generate_text_LLM import LLMClient
from src.generate_image import api_sd_generate
from src.font import Font

def create_presentation(description: str) -> str:
    """
    Generate a presentation based on the given description.
    
    Args:
        description (str): Description of the presentation to generate
    
    Returns:
        str: Path to the generated PowerPoint file
    """
    fonts_dir = "./fonts"
    logs_dir = "./logs"
    
    font = Font(fonts_dir)
    font.set_random_font() 
    
    output_dir = f'{logs_dir}/{int(time.time())}'

    # Initialize LLM Client
    llm_client = LLMClient(model_version="llama-3.1-8b-instant")
    
    generate_presentation(
        llm_generate=llm_client.generate, 
        generate_image=api_sd_generate,
        prompt_config=en_gigachat_config, 
        description=description,
        font=font,
        output_dir=output_dir,
    )

    return f'{output_dir}/presentation.pptx'

def main():
    # Example description
    input_description = "Create a presentation on electric vehicles."
    
    # Generate the presentation and get the file path
    presentation_file = create_presentation(input_description)
    
    print(f"Presentation generated: {presentation_file}")

if __name__ == "__main__": 
    main()