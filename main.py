from typing import Dict

from flask import Flask, request, jsonify

import time
import re
import cv2
import csv
import face_recognition
import numpy as np

from PIL import Image
from google.oauth2 import service_account
from google.cloud import vision

app = Flask(__name__)

credentials = service_account.Credentials.from_service_account_file(
    './edpa301-banknote-auth.json')
vision_client = vision.ImageAnnotatorClient(credentials=credentials)


@app.route('/', methods=['GET'])
def on_request_example():
    return "Hello world!"


def detect_mandela(image_path):
    image = face_recognition.load_image_file(image_path)

    # Find all face locations and face encodings in the image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Load a reference image to recognize
    reference_image_path = './mandela.jpg'
    reference_image = face_recognition.load_image_file(reference_image_path)

    reference_encoding = face_recognition.face_encodings(reference_image)[0]

    for (top, right, bottom,
         left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches the reference image
        matches = face_recognition.compare_faces([reference_encoding],
                                                 face_encoding)

        if matches[0]:
            return True

        return False

    return False


def find_serial_number(detected_text):
    # Join all extracted text into one large string
    if detected_text is None:
        return False

    # Define the string to search for
    serial_number_pattern = r'[A-Z] [A-Z] \d{8} [A-Z]'

    # Count occurrences using regex
    serial_numbers = re.findall(serial_number_pattern, detected_text)

    return serial_numbers


def detect_text_using_google_vision(image_content):
    """
    Perform text detection using Google Vision API
    Returns:
        detected_text: str | None
    """
    image = vision.Image(content=image_content)
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations
    detected_text = str(texts[0].description) if texts else None

    return detected_text


def determine_value_from_text_detection(detected_text):
    """
    Use Regex to determine the possible value of banknote
    Returns:
        note_value: "10" | "20" | "50" | "100" | "200" | None
    """
    # Define regular expression patterns for each denomination
    patterns = {
        "10": r"\b10\b",
        "20": r"\b20\b",
        "50": r"\b50\b",
        "100": r"\b100\b",
        "200": r"\b200\b"
    }

    # Search for each denomination pattern in the detected text
    # count the number of occurrences and the most frequent is the value
    res = {}
    for value, pattern in patterns.items():
        res[value] = len(re.findall(pattern, detected_text))

    # Get the denomination with the highest count
    max_value = max(res.values())
    note_value = [k for k, v in res.items() if v == max_value]

    if note_value:
        return note_value[0]

    # Return None if no denomination is found
    return None


def crop_at_banknote_edges(image):
    """
    Crops the image at the banknotes edges
    Returns:
        cropped_image: ndarray
    """
    # Read the image file from the FileStorage object
    file_bytes = np.asarray(bytearray(image), dtype=np.uint8)

    image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Apply edge detection to find contours
    edges = cv2.Canny(cv2_img, 30, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (presumably the banknote)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Make a copy of the original image to draw on
    image_with_boxes = cv2_img.copy()

    # Get the bounding rectangles for all contours
    # bounding_rectangles = [cv2.boundingRect(contour) for contour in contours]

    # Compute the minimum and maximum x, y coordinates of all contours
    min_x = min(contour[:, 0, 0].min() for contour in contours)
    min_y = min(contour[:, 0, 1].min() for contour in contours)
    max_x = max(contour[:, 0, 0].max() for contour in contours)
    max_y = max(contour[:, 0, 1].max() for contour in contours)

    # Compute the bounding rectangle
    x, y, w, h = min_x, min_y, max_x - min_x, max_y - min_y

    # Draw the merged bounding box on the image
    cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Crop the image using the bounding box
    cropped_image = cv2_img[y:y + h, x:x + w]

    return cropped_image


@app.route('/authenticate_banknote', methods=['POST'])
def authenticate_banknote():
    # Get the banknote image from the request
    banknote_image = request.files['img']
    img_content = banknote_image.read()

    detected_text = detect_text_using_google_vision(img_content)

    if detected_text is None:
        return jsonify({"error": "No text detected in the image"}), 400

    serial_numbers = find_serial_number(detected_text)
    value = determine_value_from_text_detection(detected_text)

    Image.open(banknote_image).save("banknote.jpg")

    # Check if the banknote has Mandela's face
    has_mandela = detect_mandela("banknote.jpg")

    res_obj = {
        "timestamp": time.time(),
        "serial_numbers": serial_numbers,
        "has_mandela": has_mandela,
        "value": value,
        "is_authentic": (has_mandela or serial_numbers) and value is not None
    }

    # append results to csv file. create if not existent
    with open('./static/results.csv', mode='a') as results_file:
        results_writer = csv.writer(results_file,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow([
            res_obj["timestamp"], res_obj["serial_numbers"],
            res_obj["has_mandela"], res_obj["value"], res_obj["is_authentic"]
        ])

    return jsonify({
        "value": res_obj["value"],
        "is_authentic": res_obj["is_authentic"]
    })


if __name__ == "__main__":
    with open('./static/results.csv', mode='w') as results_file:
        results_writer = csv.writer(results_file,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow([
            "timestamp", "serial_numbers", "has_mandela", "value",
            "is_authentic"
        ])

    app.run(debug=True)
