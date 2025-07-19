import pytesseract
import cv2
import numpy as np
from PIL import ImageGrab

class ScreenReader:
    def __init__(self):
        # You may need to adjust this path if tesseract is not installed in the default location
        pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

    def capture_and_read(self):
        """Capture the screen and perform OCR to extract text"""
        # Capture the screen
        img = ImageGrab.grab()
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        
        # Use Tesseract to perform OCR
        text = pytesseract.image_to_string(cv_img)
        return text
