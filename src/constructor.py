from pptx import Presentation
from pptx.util import Inches
from pptx.oxml.xmlchemy import OxmlElement
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE

import random
import os
from PIL import Image
from typing import List, Callable

from .llm_utils import llm_generate_titles, llm_generate_text, llm_generate_image_prompt, llm_generate_background_prompt
from .prompt_configs import PromptConfig
from .slides import generate_slide
from .font import Font

import tqdm


def generate_presentation(
    llm_generate: Callable[[str], str],
    generate_image: Callable[[str, int, int], Image.Image],
    prompt_config: PromptConfig,
    description: str,
    font: Font, 
    output_dir: str,
) -> Presentation:
    """Generate a PowerPoint presentation based on a description using language and image models."""
    os.makedirs(os.path.join(output_dir, 'pictures'), exist_ok=True)
    presentation = Presentation()
    presentation.slide_height = Inches(9)
    presentation.slide_width = Inches(16)

    pbar = tqdm.tqdm(total=4, desc="Presentation goes brrr...")
    
    pbar.set_description("Generating titles for presentation")
    titles = llm_generate_titles(llm_generate, description, prompt_config)
    pbar.update(1)
    
    pbar.set_description("Generating text for slides")
    texts = llm_generate_text(llm_generate, description, titles, prompt_config)
    pbar.update(1)
    
    # Generate images for all slides
    pbar.set_description("Generating images for slides")
    picture_paths = []
    for t_index, (title, text) in enumerate(zip(titles, texts)):
        image_width, image_height = random.choice([(768, 1344), (1024, 1024)])
        caption_prompt = llm_generate_image_prompt(
            llm_generate, 
            description, 
            title, 
            prompt_config
        )
        picture = generate_image(
            prompt=caption_prompt, 
            width=image_width, 
            height=image_height
        )
        picture_path = os.path.join(output_dir, 'pictures', f'{t_index:06}.png')
        picture.save(picture_path)
        picture_paths.append(picture_path)
    pbar.update(1)
    
    pbar.set_description("Packing presentation")
    
    # Generate slides - all slides will have text and image
    for title, text, picture_path in zip(titles, texts, picture_paths):
        generate_slide(
            presentation=presentation,
            title=title,
            text=text,
            picture_path=picture_path,
            background_path=None,  # No backgrounds used
            font=font,
        )
    pbar.update(1)
    
    pbar.set_description("Done")
    output_path = os.path.join(output_dir, 'presentation.pptx')
    presentation.save(output_path)
    return presentation