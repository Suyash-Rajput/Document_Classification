import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def object_character_recognition(image_path):
    custom_config = '-l eng --oem 1 --psm 6'
    text = pytesseract.image_to_string(Image.open(image_path), config=custom_config, output_type=Output.DICT)
    return text['text']


