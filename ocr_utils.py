# ocr_utils.py
import cv2
import numpy as np
from PIL import Image
import pytesseract

# If using Windows and Tesseract not in PATH, set this:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def deskew_image(gray):
    coords = np.column_stack(np.where(gray > 0))
    if coords.shape[0] == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_cv(img_bgr, resize_factor=1.0):
    # Convert to gray
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # optional resize
    if resize_factor != 1.0:
        gray = cv2.resize(gray, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
    # denoise
    gray = cv2.medianBlur(gray, 3)
    # deskew
    gray = deskew_image(gray)
    # adaptive threshold
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 11)
    return th

def ocr_with_tesseract(image_path, lang='eng', psm=3, oem=3, resize_factor=1.0):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at {image_path}")
    proc = preprocess_cv(img, resize_factor=resize_factor)
    pil = Image.fromarray(proc)
    config = f'--oem {oem} --psm {psm}'
    text = pytesseract.image_to_string(pil, lang=lang, config=config)
    return text

# Optional: easyocr path
def ocr_with_easyocr(image_path, langs=['en']):
    try:
        import easyocr
    except ImportError:
        raise RuntimeError("EasyOCR not installed. pip install easyocr")
    reader = easyocr.Reader(langs, gpu=False)  # set gpu=True if you installed cuda-enabled torch
    results = reader.readtext(image_path, detail=0, paragraph=True)
    return "\n".join(results)