# [START edpa301-banknote-auth]
import re
import cv2
import json
import skimage as ski
import numpy as np

from firebase_functions import https_fn, options
from firebase_admin import initialize_app
from google.cloud import vision

initialize_app()

vision_client = vision.ImageAnnotatorClient()

@https_fn.on_request(cors=options.CorsOptions(cors_origins="*", cors_methods=["get"]))
def on_request_example(req: https_fn.Request) -> https_fn.Response:
    return https_fn.Response("Hello world!")


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
    for note_value, pattern in patterns.items():
        if re.search(pattern, detected_text, flags=re.IGNORECASE):
            return note_value

    # Return None if no denomination is found
    return None


def crop_at_banknote_edges(image):
    """
    Crops the image at the banknotes edges
    Returns:
        cropped_image: ndarray
    """
    # Read the image file from the FileStorage object
    # image_stream = image.stream
    file_bytes = np.asarray(bytearray(image), dtype=np.uint8)
    
    image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Apply edge detection to find contours
    edges = cv2.Canny(cv2_img, 30, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (presumably the banknote)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Make a copy of the original image to draw on
    image_with_boxes = cv2_img.copy()

    # Get the bounding rectangles for all contours
    bounding_rectangles = [cv2.boundingRect(contour) for contour in contours]

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
    cropped_image = cv2_img[y:y+h, x:x+w]

    return cropped_image


def enhance_features(image):
    """
    Uses skimage to:
    Increases highlights,
    Increases brightness,
    Decreases exposure,
    Increases contrast
    """
    p2, p98 = np.percentile(image, (2, 98))
    image = ski.exposure.rescale_intensity(image, in_range=(p2, p98))
    image = ski.exposure.adjust_gamma(image, gamma=2)
    image = ski.exposure.adjust_sigmoid(image, cutoff=0.5, gain=8)

    return image


def create_histogram(image):
    """
    Creates a histogram
    Returns:
        hist: array
        bin_edges: array of dtype float
    """
    img_float_ndarray = ski.util.img_as_float(image)
    img_float_ndarray[img_float_ndarray < 0.4] = -1
    histogram, bin_edges = np.histogram(img_float_ndarray, bins=256, range=(0, 1))
    
    return (histogram, bin_edges)


@https_fn.on_request(cors=options.CorsOptions(cors_origins="*", cors_methods=["post"]))
def authenticate_banknote(req: https_fn.Request) -> https_fn.Response:
    """Authenticate a banknote image sent via HTTP POST."""
    # Get the banknote image from the request
    banknote_image = req.files['img']
    img_content = banknote_image.read()

    detected_text = detect_text_using_google_vision(img_content)

    if detected_text is None:
        return https_fn.Response("No text detected in the image", status=400)

    value = determine_value_from_text_detection(detected_text)
    cropped_img = crop_at_banknote_edges(img_content)

    feature_enhanced_image = enhance_features(cropped_img)
    hist, bin_edges = create_histogram(feature_enhanced_image)

    res_obj = {
        "value": value,
        "detected_text": detected_text,
        "res": None
    }

    return https_fn.Response(json.dumps(str(res_obj)))

# [END edpa301-banknote-auth]