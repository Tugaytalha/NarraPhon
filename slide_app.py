# app.py - WE DON'T NEED THIS FILE SINCE IT IS THE INTERFACE
import gradio as gr
import time
from src.constructor import generate_presentation 
from src.prompt_configs import en_gigachat_config
from src.generate_text_LLM import giga_generate
from src.generate_image import api_k31_generate
from src.font import Font

logs_dir = "logs"
fonts_dir = "fonts"

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
        generate_image=api_k31_generate,
        prompt_config=en_gigachat_config, 
        description=description,
        font=font,
        output_dir=output_dir,
    )

    return f'{output_dir}/presentation.pptx'

# Updated examples to include language selection
examples = [
    ["Generate a presentation on economics, 7 slides", "English"],
    ["Сгенерируйте презентацию по экономике, 7 слайдов", "Русский"],
    ["Create a presentation on climate change, 6 slides", "English"],
    ["Создайте презентацию об изменении климата, 6 слайдов", "Русский"],
    ["Create a presentation on artificial intelligence, 8 slides", "English"],
    ["Создайте презентацию об искусственном интеллекте, 8 слайдов", "Русский"],
    ["Design a presentation on space exploration, 10 slides", "English"],
    ["Разработайте презентацию о космических исследованиях, 10 слайдов", "Русский"],
    ["Prepare a presentation on the future of renewable energy, 7 slides", "English"],
    ["Подготовьте презентацию о будущем возобновляемой энергетики, 7 слайдов", "Русский"],
    ["Develop a presentation on the history of art movements, 9 slides", "English"],
    ["Разработайте презентацию о истории художественных движений, 9 слайдов", "Русский"],
    ["Generate a presentation on the impact of social media, 6 slides", "English"],
    ["Сгенерируйте презентацию о влиянии социальных сетей, 6 слайдов", "Русский"],
    ["Create a presentation on sustainable urban planning, 8 slides", "English"],
    ["Создайте презентацию о устойчивом градостроительстве, 8 слайдов", "Русский"],
    ["Разработайте презентацию о новшествах в области медицинских технологий, 7 слайдов", "Русский"],
    ["Design a presentation on innovations in healthcare technology, 7 slides", "English"],
    ["Подготовьте презентацию о глобальных экономических тенденциях, 5 слайдов", "Русский"],
    ["Prepare a presentation on global economic trends, 5 slides", "English"],
    ["Разработайте презентацию о психологии потребительского поведения, 6 слайдов", "Русский"],
    ["Develop a presentation on the psychology of consumer behavior, 6 slides", "English"],
    ["Сгенерируйте презентацию о преимуществах осознанности и медитации, 7 слайдов", "Русский"],
    ["Generate a presentation on the benefits of mindfulness and meditation, 7 slides", "English"],
    ["Создайте презентацию о достижениях в области автономных транспортных средств, 8 слайдов", "Русский"],
    ["Create a presentation on advancements in autonomous vehicles, 8 slides", "English"],
    ["Разработайте презентацию о влиянии изменений климатической политики, 5 слайдов", "Русский"],
    ["Design a presentation on the impact of climate policy changes, 5 slides", "English"],
]

iface = gr.Interface(
    fn=create_presentation,
    inputs=[
        gr.Textbox(
            label="Presentation Description", 
            placeholder="Enter the description for the presentation..."
        ),
        gr.Dropdown(
            label="Language",
            choices=["English", "Russian"],
            value="English"
        )
    ],
    outputs=gr.File(
        label="Download Presentation"
    ),
    title="Presentation Generator",
    description="Generate a presentation based on the provided description and selected language. Click the button to download the presentation.",
    css="footer {visibility: hidden}",
    allow_flagging="never",  
    examples=examples  
)

iface.launch()