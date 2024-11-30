from typing import List

prefix = "prompt: "

class PromptConfig:
    def __init__(
        self, 
        title_prompt: str, 
        text_prompt: str, 
        image_prompt: str, 
        background_prompt: str,
        background_styles: List[str],
    ):
        self.title_prompt = title_prompt
        self.text_prompt = text_prompt
        self.image_prompt = image_prompt
        self.background_prompt = background_prompt
        self.background_styles = background_styles