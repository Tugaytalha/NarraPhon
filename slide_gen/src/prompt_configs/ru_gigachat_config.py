from .prompt_config import PromptConfig, prefix

ru_gigachat_config = PromptConfig(
    title_prompt = (
        'тебе дано описание презентации: "{description}". '
        'На основе данного описания и примеров, сгенерируй заголовки слайдов презентации. '
        'Заголовок должен быть коротким, не более 4 слов. '
        'Представь ответ в виде пронумерованного списка. '
        'Примеры:\n '
        'Запрос: Описание презентации о стратегии маркетинга для нового продукта.\n'
        '1. Введение\n '
        '2. Цели маркетинга\n '
        '3. Анализ рынка\n '
        '4. Бюджет\n '
        '5. Заключение\n '
        'Запрос: Презентация о достижениях компании за прошлый год.\n'
        '1. Приветствие\n '
        '2. Общие достижения\n '
        '3. Финансовые результаты\n '
        '4. Успешные проекты\n '
        '5. Развитие команды\n '
        '6. Социальные инициативы\n '
        '7. Планы на будущее\n '
        '8. Благодарности\n '
        '9. Вопросы и ответы\n '
        'Запрос: Презентация о новых технологиях в производстве.\n'
        '1. Введение в тему\n '
        '2. Текущие технологии\n '
        '3. Новые разработки\n '
        '4. Примеры внедрения\n '
        '5. Будущие тенденции\n '
        '6. Заключение\n '
        '7. Дискуссия\n '
        'Ответ:\n'
    ),
    text_prompt = (
        'тебе дано описание презентации: "{description}". '
        'Напиши одно предложение не более 20 слов для слайда с заголовком "{title}". '
        f'Напиши только итоговый текст, начинай с "{prefix} ". '
        'Примеры:\n'
        f'{prefix} Увеличение продаж на 20% связано с внедрением новой маркетинговой стратегии.\n'
        f'{prefix} Инновационные технологии помогли повысить эффективность производства на 30%.\n'
        f'{prefix} Новые подходы к работе с клиентами увеличили уровень удовлетворенности на 15%.\n'
        f'{prefix} В этом году компания запустила три новых продукта, которые стали лидерами на рынке.\n'
        'Ответ:\n'
    ),
    image_prompt = (
        'тебе дано описание презентации: "{description}". '
        'Придумай детализированное описание эстетичной картинки для слайда с заголовком: "{title}". '
        'Описание должно быть длинным и супер детализированным, включающим все аспекты визуальной составляющей. '
        'Исключи цифровые значения, текст, графики, названия компаний и тому подобное. '
        'Избегай использования текста на изображении. '
        'Сделай его визуально приятным и подходящим контексту. '
        'Начни со слова "описание: ". '
        'Примеры:\n'
        f'{prefix} Просторный зал заседаний с современным дизайном, стеклянные стены пропускают много естественного света, в центре длинный деревянный стол с ноутбуками и документами, вокруг сидят деловые люди в официальной одежде, на заднем плане видна городская панорама через окна.\n'
        f'{prefix} Лесная тропа, окруженная высокими деревьями с зелеными листьями, на земле опавшая листва, солнечные лучи пробиваются сквозь листву, создавая игру света и теней, на тропе видны следы животных, вдали слышен шум реки.\n'
        f'{prefix} Оживленная улица в центре города, по обе стороны высокие современные здания со стеклянными фасадами, на улице много прохожих, некоторые спешат, другие медленно прогуливаются, между ними едут автомобили и автобусы, небо ясное с редкими облаками.\n'
        f'{prefix} Уютное кафе с деревянными столами и мягкими креслами, на стенах висят картины с изображением природы, большие окна пропускают много света, за столами сидят посетители, некоторые работают за ноутбуками, другие беседуют за чашкой кофе, на стойке видны десерты и напитки.\n'
        'Ответ:\n'
    ),
    background_prompt = (
        'На основании описания презентации: {description} '
        'и заголовка текущего слайда: "{title}". '
        'Используй in-context learning для генерации 4 ключевых слов. '
        'Напиши их через запятую. '
        'Примеры:\n'
        'инновации, рост, технологии, успех\n'
        'экология, устойчивость, природа, будущее\n'
        'развитие, обучение, достижения, цели\n'
        'ответственность, сообщество, проекты, партнерство\n'
        'Ответ:\n'
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