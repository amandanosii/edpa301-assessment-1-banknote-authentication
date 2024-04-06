# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`
from firebase_functions import https_fn, options
from firebase_admin import initialize_app
from google.cloud import vision

import skimage as ski
import numpy as np

# Initialize Firebase App
initialize_app()

# Initialize Google Cloud Vision API client
vision_client = vision.ImageAnnotatorClient()

@https_fn.on_request(cors=options.CorsOptions(cors_origins="*", cors_methods=["get"]))
def on_request_example(req: https_fn.Request) -> https_fn.Response:
    return https_fn.Response("Hello world!")


def crop_at_banknote_edges(image):
    """
    Crops the image at the banknotes edges
    """
    return


def enhance_figures(image):
    """
    Uses skimage to:
    Increases brightness, 
    Decreases exposure, 
    Increases contrast,
    Increases highlights,
    Decreases shadows
    """
    return


def create_histogram(image):
    """
    Creates a histogram
    Returns:
        hist: array
        bin_edges: array of dtype float
    """
    img_float_ndarray = ski.util.img_as_float(image)
    histogram, bin_edges = np.histogram(img_float_ndarray, bins=256, range=(0, 1))
    return (histogram, bin_edges)

def detect_text_using_google_vision(banknote_image):
    """
    Perform text detection using Google Vision API
    Returns:
        detected_text: str | None
    """
    image_content = banknote_image.read()
    image = vision.Image(content=image_content)
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations
    detected_text = str(texts[0].description) if texts else None

    return detected_text


def determine_value_from_text_detection(detected_text):
    """
    Use Regex to determine the possible value of banknote
    Returns:
        note_value: "10" | "20" | "50" | "100" | "200"
    """
    return


@https_fn.on_request(cors=options.CorsOptions(cors_origins="*", cors_methods=["post"]))
def authenticate_banknote(req: https_fn.Request) -> https_fn.Response:
    """Authenticate a banknote image sent via HTTP POST."""
    # Get the banknote image from the request
    banknote_image = req.files['img']

    detected_text = detect_text_using_google_vision(banknote_image)
    
    if detected_text is None:
        return https_fn.Response("No text detected in the image", status=400)

    return https_fn.Response(detected_text)
