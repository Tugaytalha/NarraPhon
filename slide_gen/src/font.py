import os
import random
from typing import Optional

class Font:
    def __init__(self, fonts_dir: str, max_size: int = 66):
        """
        Initialize the Font class with a directory containing font files.

        Args:
            fonts_dir (str): Path to the directory containing font files.
            max_size (int): Maximum font size to use for fitting text.
        """
        self.fonts_dir = fonts_dir
        self.font_name = None  # Default font
        self.set_random_font()
        self.max_size = max_size

    def set_font(self, font_name: str = "Tahoma") -> None:
        """
        Set the font name to be used.

        Args:
            font_name (str): Name of the font to set (default is "Tahoma").
        """
        if self._find_font(font_name):
            self.font_name = font_name
        else:
            raise ValueError(f"Font '{font_name}' not found in '{self.fonts_dir}'.")

    def set_random_font(self) -> None:
        """
        Set a random font from the fonts directory. The chosen font must have both
        basic and bold styles available.
        """
        available_fonts = self._find_available_fonts()
        if not available_fonts:
            raise ValueError("No fonts with both basic and bold styles found.")

        self.font_name = random.choice(available_fonts)

    @property
    def basic(self) -> Optional[str]:
        """
        Get the path of the basic font style based on the current font name.

        Returns:
            Optional[str]: The full path to the basic font style or None if not found.
        """
        return self._find_font(f'{self.font_name}')

    @property
    def bold(self) -> Optional[str]:
        """
        Get the path of the bold font style based on the current font name.

        Returns:
            Optional[str]: The full path to the bold font style or None if not found.
        """
        return self._find_font(f'{self.font_name}Bd')

    @property
    def italic(self) -> Optional[str]:
        """
        Get the path of the italic font style based on the current font name.

        Returns:
            Optional[str]: The full path to the italic font style or None if not found.
        """
        return self._find_font(f'{self.font_name}It')

    @property
    def italic_bold(self) -> Optional[str]:
        """
        Get the path of the italic bold font style based on the current font name.

        Returns:
            Optional[str]: The full path to the italic bold 
                        font style or None if not found.
        """
        return self._find_font(f'{self.font_name}BdIt')

    def _find_font(self, font_name: str) -> Optional[str]:
        """
        Find a font file in the fonts directory by font name.

        Args:
            font_name (str): The font name to find.

        Returns:
            Optional[str]: The full path to the font file if found, None otherwise.
        """
        if not font_name.endswith(".ttf"):
            font_name = f'{font_name}.ttf'
            
        for filename in os.listdir(self.fonts_dir):
            if font_name == filename:
                file_path = os.path.join(self.fonts_dir, filename)
                return file_path
        return None

    def _find_available_fonts(self) -> list:
        """
        Find all available fonts in the fonts directory 
        that have both basic and bold styles.

        Returns:
            list: A list of font names (without file extension) 
                    that have both basic and bold styles.
        """
        fonts = set()
        for filename in os.listdir(self.fonts_dir):
            if filename.endswith(".ttf"):
                font_name = filename[:-4]  # Remove the .ttf extension
                if font_name.endswith("Bd"):
                    basic_font = font_name[:-2]
                    if os.path.exists(os.path.join(self.fonts_dir, f"{basic_font}.ttf")):
                        fonts.add(basic_font)
                else:
                    bold_font = f"{font_name}Bd"
                    if os.path.exists(os.path.join(self.fonts_dir, f"{bold_font}.ttf")):
                        fonts.add(font_name)
        return list(fonts)