from .prompt_config import PromptConfig, prefix

en_gigachat_config = PromptConfig(
    title_prompt = (
        'You are given a presentation description: "{description}". '
        'Based on this description and examples, generate slide titles for the presentation. '
        'The title should be brief, no more than 4 words. '
        'Answer in English only. '
        'Present the response as a numbered list. '
        'Examples:\n '
        'Query: Description of a presentation about marketing strategy for a new product.\n'
        '1. Introduction\n '
        '2. Marketing Goals\n '
        '3. Market Analysis\n '
        '4. Budget\n '
        '5. Conclusion\n '
        'Query: Presentation about company achievements over the past year.\n'
        '1. Welcome\n '
        '2. General Achievements\n '
        '3. Financial Results\n '
        '4. Successful Projects\n '
        '5. Team Development\n '
        '6. Social Initiatives\n '
        '7. Future Plans\n '
        '8. Acknowledgments\n '
        '9. Q&A\n '
        'Query: Presentation about new technologies in manufacturing.\n'
        '1. Introduction\n '
        '2. Current Technologies\n '
        '3. New Developments\n '
        '4. Implementation Examples\n '
        '5. Future Trends\n '
        '6. Conclusion\n '
        '7. Discussion\n '
        'Response:\n'
    ),
    text_prompt = (
        'You are given a presentation description: "{description}". '
        'Write one sentence no more than 20 words for a slide with the title "{title}". '
        'Answer in English only. '
        f'Write only the final text, starting with "{prefix} ". '
        'Examples:\n'
        f'{prefix} The 20% sales increase is attributed to the implementation of the new marketing strategy.\n'
        f'{prefix} Innovative technologies have improved manufacturing efficiency by 30%.\n'
        f'{prefix} New customer engagement approaches have increased satisfaction levels by 15%.\n'
        f'{prefix} This year, the company launched three new products that became market leaders.\n'
        'Response:\n'
    ),
    image_prompt = (
        'You are given a presentation description: "{description}". '
        'Generate a detailed description of an aesthetic image for a slide with the title: "{title}". '
        'The description should be long and highly detailed, covering all aspects of the visual elements. '
        'Exclude numerical values, text, graphs, company names, and similar content. '
        'Avoid using text on the image. '
        'Answer in English only. '
        'Make it visually pleasing and contextually appropriate. '
        'Start with the word "Description: ". '
        'Examples:\n'
        f'{prefix} A spacious conference room with a modern design, glass walls letting in plenty of natural light, a long wooden table in the center with laptops and documents, business people in formal attire sitting around, and a cityscape visible through the windows.\n'
        f'{prefix} A forest trail surrounded by tall trees with green leaves, fallen leaves on the ground, sunlight filtering through the foliage creating a play of light and shadow, animal tracks visible on the path, and the distant sound of a river.\n'
        f'{prefix} A busy street in the city center, with high modern buildings featuring glass facades on both sides, many pedestrians walking, some rushing and others strolling, cars and buses moving along the street, and a clear sky with a few clouds.\n'
        f'{prefix} A cozy café with wooden tables and soft chairs, paintings of nature on the walls, large windows letting in plenty of light, patrons sitting at tables, some working on laptops and others chatting over coffee, and a counter with desserts and beverages.\n'
        'Response:\n'
    ),
    background_prompt = (
        'Based on the presentation description: "{description}" '
        'and the current slide title: "{title}". '
        'Use in-context learning to generate 4 key words related to the content of the slide. '
        'Write the key words separated by commas. '
        'Examples:\n'
        'Input: Presentation about the latest trends in digital marketing.\n'
        'Title: Emerging Technologies\n'
        f'{prefix} innovation, digital, trends, technology\n\n'
        'Input: Presentation on strategies for improving customer service.\n'
        'Title: Enhancing Engagement\n'
        f'{prefix} customer, engagement, strategies, improvement\n\n'
        'Input: Presentation on the impact of climate change on agriculture.\n'
        'Title: Environmental Challenges\n'
        f'{prefix} climate, agriculture, impact, sustainability\n\n'
        'Input: Presentation on the benefits of remote work for productivity.\n'
        'Title: Work Efficiency\n'
        f'{prefix} remote, productivity, benefits, efficiency\n'
        'Response:\n'
    ),
    # List of strings!!!
    background_styles = [
        (
            'Gradient. WITHOUT TEXT, Vectors style, '
            'Gradient dip, More game with colors, Smooth transition. '
        ),
        (
            'Abstract. Clean lines, Modern feel, '
            'Minimalistic, Soft colors, Elegant look. '
        ),
        (
            'Nature-inspired. Soft green tones, '
            'Earthy feel, Natural textures, Organic look. '
        ),
        (
            'Technology. Futuristic design, Blue tones, '
            'Circuit patterns, Sleek lines, High-tech feel. '
        ),
        (
            'Corporate. Professional look, Subtle gradients, '
            'Clean and polished, Neutral colors, Business-oriented. '
        ),
        (
            'Retro. Bold colors, Geometric shapes, '
            'Vintage feel, Nostalgic design, Playful patterns. '
        ),
        (
            'Minimalist. White space, Simple shapes, '
            'Clean and clear, Monochrome tones, Modern elegance. '
        ),
        (
            'Art Deco. Rich textures, Metallic accents, '
            'Geometric patterns, Glamorous style, 1920s influence. '
        ),
        (
            'Urban. Graffiti art, Vibrant colors, '
            'Street style, Dynamic patterns, Energetic vibe. '
        ),
        (
            'Watercolor. Soft brush strokes, Blended hues, '
            'Artistic feel, Fluid shapes, Subtle transitions. '
        ),
        (
            'Dark Mode. Deep black tones, Subtle contrasts, '
            'Sophisticated look, Modern design, High contrast elements. '
        ),
        (
            'Elegant. Rich colors, Decorative patterns, '
            'Luxurious textures, Classic style, Refined details. '
        ),
        (
            'Nature-inspired. Earthy colors, Leaf patterns, '
            'Wood textures, Tranquil feel, Organic shapes. '
        ),
        (
            'Dynamic. Bold contrasts, Energetic lines, '
            'Motion feel, Vibrant colors, Modern design. '
        )
    ]
)