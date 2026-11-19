
"""
Lab Assignment 9 - Cognitive Computing (UCS420)
Project 1: Medicine Label Reader

Name: Madhav Narang
Roll No.: 1024170287
"""

import re
import argparse
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter
import pytesseract


def preprocess_image(image_path):
    """Improve image quality before OCR."""
    image = Image.open(image_path).convert("RGB")
    image = ImageEnhance.Contrast(image).enhance(1.8)
    image = ImageEnhance.Sharpness(image).enhance(2.0)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    return image


def extract_fields(text):
    """Extract important medicine information."""

    patterns = {
        "Medicine Name": [
            r"(?:brand\s*name|medicine\s*name|product\s*name)\s*[:\-]\s*(.+)"
        ],
        "Strength": [
            r"(?:strength|composition)\s*[:\-]?\s*(.+)",
            r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|%|IU)\b"
        ],
        "Dosage": [
            r"(?:dosage|directions|how\s*to\s*use|take)\s*[:\-]\s*(.+)"
        ],
        "Expiry Date": [
            r"(?:exp(?:iry)?\.?\s*date|expires?)\s*[:\-]?\s*([A-Za-z0-9/\-]+)",
            r"\b(?:0[1-9]|1[0-2])[/\-](?:20)?\d{2}\b"
        ],
        "Batch Number": [
            r"(?:batch\s*(?:no\.?|number)?|lot\s*(?:no\.?|number)?)\s*[:\-]\s*([A-Za-z0-9\-]+)"
        ],
        "Manufacturer": [
            r"(?:manufactured\s*by|mfg\.?\s*by|manufacturer)\s*[:\-]\s*(.+)"
        ],
        "Warnings": [
            r"(?:warning|caution|storage)\s*[:\-]?\s*(.+)"
        ]
    }

    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]

    results = {}

    for field, field_patterns in patterns.items():
        results[field] = "Not detected"

        for pattern in field_patterns:
            for line in lines:
                match = re.search(pattern, line, re.IGNORECASE)

                if match:
                    results[field] = (
                        match.group(1).strip()
                        if match.lastindex
                        else match.group(0)
                    )
                    break

            if results[field] != "Not detected":
                break

    return results


def read_medicine_label(image_path):

    if not Path(image_path).is_file():
        raise FileNotFoundError("Image file not found!")

    image = preprocess_image(image_path)

    text = pytesseract.image_to_string(
        image,
        config="--psm 6"
    )

    information = extract_fields(text)

    return text, information


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        help="Path of medicine label image"
    )

    args = parser.parse_args()

    try:
        raw_text, information = read_medicine_label(args.image)

    except pytesseract.TesseractNotFoundError:
        print("Tesseract OCR is not installed.")
        print("Install it using: brew install tesseract")
        return

    except Exception as error:
        print("Error:", error)
        return

    print("\n========================================")
    print("       MEDICINE LABEL READER")
    print("========================================")

    print("Name: Madhav Narang")
    print("Roll No.: 1024170287")

    print("\nEXTRACTED MEDICINE INFORMATION")

    for key, value in information.items():
        print(f"{key}: {value}")

    print("\nRAW OCR TEXT")
    print("----------------------------------------")
    print(raw_text if raw_text.strip() else "No text detected.")

    print("\nSAFETY NOTICE:")
    print("Always verify medicine information")
    print("from the original packaging or a pharmacist.")
    print("Do not use OCR output to decide dosage.")


if __name__ == "__main__":
    main()
