import os
import requests

OCR_URL = "https://api.ocr.space/parse/image"
API_KEY = "K86032675288957"


def extract_text(image_path):
    with open(image_path, "rb") as f:
        response = requests.post(
            OCR_URL,
            files={"file": f},
            data={
                "language": "eng",
                "isOverlayRequired": False,
                "OCREngine": 2,        # better for handwriting
                "scale": True
            },
            headers={
                "apikey": API_KEY
            },
            timeout=30
        )

    response.raise_for_status()
    result = response.json()

    if result.get("IsErroredOnProcessing"):
        raise RuntimeError(result.get("ErrorMessage"))

    parsed = result["ParsedResults"][0]["ParsedText"]
    return parsed


if __name__ == "__main__":
    text = extract_text("hi.jpg")
    print("\n--- OCR OUTPUT ---\n")
    print(text)