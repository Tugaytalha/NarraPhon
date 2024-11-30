from typing import List, Callable

from src.prompt_configs import PromptConfig, prefix

def llm_generate_titles(
    llm_generate: Callable[[str], str], 
    description: str, 
    prompt_config: PromptConfig,
) -> List[str]:
    """
    Generate presentation slide titles using a language model.
    """
    prompt = prompt_config.title_prompt.format(
        description=description
    )
    titles_str = llm_generate(prompt)
    titles = []
    
    # Split by newline and process each title
    for title in titles_str.split("\n"):
        # Remove any leading numbers or dots
        title = title.strip()
        
        # Remove introductory text and unwanted phrases
        unwanted_phrases = [
            "slide titles:", 
            "based on the description", 
            "here are the slide titles", 
            "query:", 
            "response:"
        ]
        
        # Convert to lowercase for case-insensitive matching
        title_lower = title.lower()
        
        # Skip titles that contain unwanted phrases
        if any(phrase in title_lower for phrase in unwanted_phrases):
            continue
        
        # Remove leading numbers or dots
        try:
            # Method 1: Remove leading number and dot
            if '. ' in title:
                sep_index = title.index('. ') + 2
                title = title[sep_index:].strip()
            elif '.' in title:
                sep_index = title.index('.') + 1
                title = title[sep_index:].strip()
        except ValueError:
            pass
        
        # Remove any remaining punctuation and clean up
        title = title.replace('.', '').replace('\n', '').strip()
        
        # Skip empty titles, generic text, or very short titles
        if (title and 
            title.lower() not in ['title', 'slide titles'] and 
            len(title) > 2):
            titles.append(title)
    
    # Ensure we have at least a few titles
    if len(titles) < 3:
        print("Warning: Few titles generated. Using default titles.")
        titles = ["Introduction", "Main Content", "Conclusion"]
    
    return titles

def llm_generate_text(
    llm_generate: Callable[[str], str], 
    description: str, 
    titles: List[str], 
    prompt_config: PromptConfig
) -> List[tuple[str, str]]:
    """
    Generate text and speaker notes for each slide title using a language model.

    Returns:
        List[tuple[str, str]]: List of (slide_text, speaker_notes) pairs for each slide.
    """
    texts_and_notes = []
    for title in titles:
        # Generate slide text
        text_query = prompt_config.text_prompt.format(description=description, title=title)
        text = llm_generate(text_query)
        if prefix in text.lower():
            text = text[text.lower().index(prefix)+len(prefix):]
            text = text.replace('\n', '')

        # Generate speaker notes
        notes_query = f"Generate speaker notes for the slide titled '{title}'. Do not include introductory sentences like 'content may include, speaker notes may include etc.'. The notes should expand on the slide content, providing additional context and information in continuous text format. Avoid instructions or suggestions for speaking or presenting. Do not keep it too long."
        notes = llm_generate(notes_query)
        if prefix in notes.lower():
            notes = notes[notes.lower().index(prefix)+len(prefix):]
            notes = notes.replace('\n', '')

        texts_and_notes.append((text, notes))
    return texts_and_notes

def llm_generate_image_prompt(
    llm_generate: Callable[[str], str], 
    description: str, 
    title: str, 
    prompt_config: PromptConfig
) -> str:
    """
    Generate an image prompt for a slide using a language model.

    Args:
        llm_generate (Callable[[str], str]): Function to generate text using a language model.
        description (str): Description of the presentation.
        title (str): Slide title.
        prompt_config (PromptConfig): Configuration for prompts.

    Returns:
        str: Image prompt.
    """
    query = prompt_config.image_prompt.format(description=description, title=title)
    prompt = llm_generate(query)
    if prefix in prompt: 
        prompt = prompt[prompt.lower().index(prompt)+len(prompt):]
        prompt = prompt.replace('\n', '')
    return prompt

def llm_generate_background_prompt(
    llm_generate: Callable[[str], str], 
    description: str, 
    title: str, 
    prompt_config: PromptConfig, 
    background_style: str = ''
) -> str:
    """
    Generate a background prompt for a slide using a language model.

    Args:
        llm_generate (Callable[[str], str]): Function to generate text using a language model.
        description (str): Description of the presentation.
        title (str): Slide title.
        prompt_config (PromptConfig): Configuration for prompts.

    Returns:
        str: Background prompt.
    """
    query = prompt_config.background_prompt.format(description=description, title=title)
    
    keywords = llm_generate(query)
    background_prompt = f'{keywords}, {background_style}'
        
    return background_prompt