# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`
from datetime import datetime
from firebase_functions import https_fn, options
from firebase_admin import initialize_app
from google.cloud import vision

# Initialize Firebase App
initialize_app()

# Initialize Google Cloud Vision API client
vision_client = vision.ImageAnnotatorClient()

@https_fn.on_request(cors=options.CorsOptions(cors_origins="*", cors_methods=["get"]))
def on_request_example(req: https_fn.Request) -> https_fn.Response:
    return https_fn.Response("Hello world!")

@https_fn.on_request(cors=options.CorsOptions(cors_origins="*", cors_methods=["post"]))
def authenticate_banknote(req: https_fn.Request) -> https_fn.Response:
    """Authenticate a banknote image sent via HTTP POST."""
    # Get the banknote image from the request
    banknote_image = req.files['img']

    # Perform text detection using Google Vision API
    image_content = banknote_image.read()
    image = vision.Image(content=image_content)
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations
    detected_text = texts[0].description if texts else None
    
    if detected_text is None:
        return https_fn.Response("No text detected in the image", status=400)

    return https_fn.Response(detected_text)
