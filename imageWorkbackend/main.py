## geopy library install
from geopy.geocoders import Nominatim
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps, ExifTags
import numpy as np
#from teachable_machine import TeachableMachine
import json
from tensorflow.keras.models import model_from_json
from roboflow import Roboflow
import uuid
import cv2

# Mango subcategory A
MANGO_SUBCATEGORY_Mango_A_COLOUR_MODEL_PATH = 'models/mango_colour_model.h5'
MANGO_SUBCATEGORY_Mango_A_COLOUR_MODEL_LABELS_PATH = 'models/mango_colour_model_labels.txt'
MANGO_SUBCATEGORY_Mango_A_SIZE_MODEL_PATH = 'models/mango_size_model.h5'
MANGO_SUBCATEGORY_Mango_A_SIZE_MODEL_LABELS_PATH = 'models/mango_size_model_labels.txt'
MANGO_SUBCATEGORY_Mango_A_SHAPE_MODEL_PATH = 'models/mango_shape_model.h5'
MANGO_SUBCATEGORY_Mango_A_SHAPE_MODEL_LABELS_PATH = 'models/mango_shape_model_labels.txt'
MANGO_SUBCATEGORY_Mango_A_SURFACE_MODEL_PATH = 'models/mango_surface_model.h5'
MANGO_SUBCATEGORY_Mango_A_SURFACE_MODEL_LABELS_PATH = 'models/mango_surface_model_labels.txt'

# Mango subcategory B
MANGO_SUBCATEGORY_Mango_B_COLOUR_MODEL_PATH = 'models/mango_colour_model.h5'
MANGO_SUBCATEGORY_Mango_B_COLOUR_MODEL_LABELS_PATH = 'models/mango_colour_model_labels.txt'
MANGO_SUBCATEGORY_Mango_B_SIZE_MODEL_PATH = 'models/mango_size_model.h5'
MANGO_SUBCATEGORY_Mango_B_SIZE_MODEL_LABELS_PATH = 'models/mango_size_model_labels.txt'
MANGO_SUBCATEGORY_Mango_B_SHAPE_MODEL_PATH = 'models/mango_shape_model.h5'
MANGO_SUBCATEGORY_Mango_B_SHAPE_MODEL_LABELS_PATH = 'models/mango_shape_model_labels.txt'
MANGO_SUBCATEGORY_Mango_B_SURFACE_MODEL_PATH = 'models/mango_surface_model.h5'
MANGO_SUBCATEGORY_Mango_B_SURFACE_MODEL_LABELS_PATH = 'models/mango_surface_model_labels.txt'

# Mango subcategory C
MANGO_SUBCATEGORY_Mango_C_COLOUR_MODEL_PATH = 'models/mango_colour_model.h5'
MANGO_SUBCATEGORY_Mango_C_COLOUR_MODEL_LABELS_PATH = 'models/mango_colour_model_labels.txt'
MANGO_SUBCATEGORY_Mango_C_SIZE_MODEL_PATH = 'models/mango_size_model.h5'
MANGO_SUBCATEGORY_Mango_C_SIZE_MODEL_LABELS_PATH = 'models/mango_size_model_labels.txt'
MANGO_SUBCATEGORY_Mango_C_SHAPE_MODEL_PATH = 'models/mango_shape_model.h5'
MANGO_SUBCATEGORY_Mango_C_SHAPE_MODEL_LABELS_PATH = 'models/mango_shape_model_labels.txt'
MANGO_SUBCATEGORY_Mango_C_SURFACE_MODEL_PATH = 'models/mango_surface_model.h5'
MANGO_SUBCATEGORY_Mango_C_SURFACE_MODEL_LABELS_PATH = 'models/mango_surface_model_labels.txt'

# Apple subcategory A
APPLE_SUBCATEGORY_Apple_A_COLOUR_MODEL_PATH = 'models/apple_colour_model.h5'
APPLE_SUBCATEGORY_Apple_A_COLOUR_MODEL_LABELS_PATH = 'models/apple_colour_model_labels.txt'
APPLE_SUBCATEGORY_Apple_A_SIZE_MODEL_PATH = 'models/apple_size_model.h5'
APPLE_SUBCATEGORY_Apple_A_SIZE_MODEL_LABELS_PATH = 'models/apple_size_model_labels.txt'
APPLE_SUBCATEGORY_Apple_A_SHAPE_MODEL_PATH = 'models/apple_shape_model.h5'
APPLE_SUBCATEGORY_Apple_A_SHAPE_MODEL_LABELS_PATH = 'models/apple_shape_model_labels.txt'

# Apple subcategory B
APPLE_SUBCATEGORY_Apple_B_COLOUR_MODEL_PATH = 'models/apple_colour_model.h5'
APPLE_SUBCATEGORY_Apple_B_COLOUR_MODEL_LABELS_PATH = 'models/apple_colour_model_labels.txt'
APPLE_SUBCATEGORY_Apple_B_SIZE_MODEL_PATH = 'models/apple_size_model.h5'
APPLE_SUBCATEGORY_Apple_B_SIZE_MODEL_LABELS_PATH = 'models/apple_size_model_labels.txt'
APPLE_SUBCATEGORY_Apple_B_SHAPE_MODEL_PATH = 'models/apple_shape_model.h5'
APPLE_SUBCATEGORY_Apple_B_SHAPE_MODEL_LABELS_PATH = 'models/apple_shape_model_labels.txt'

# Apple subcategory C
APPLE_SUBCATEGORY_Apple_C_COLOUR_MODEL_PATH = 'models/apple_colour_model.h5'
APPLE_SUBCATEGORY_Apple_C_COLOUR_MODEL_LABELS_PATH = 'models/apple_colour_model_labels.txt'
APPLE_SUBCATEGORY_Apple_C_SIZE_MODEL_PATH = 'models/apple_size_model.h5'
APPLE_SUBCATEGORY_Apple_C_SIZE_MODEL_LABELS_PATH = 'models/apple_size_model_labels.txt'
APPLE_SUBCATEGORY_Apple_C_SHAPE_MODEL_PATH = 'models/apple_shape_model.h5'
APPLE_SUBCATEGORY_Apple_C_SHAPE_MODEL_LABELS_PATH = 'models/apple_shape_model_labels.txt'

# Strawberry subcategory A
STRAWBERRY_SUBCATEGORY_Strawberry_A_COLOUR_MODEL_PATH = 'models/strawberry_colour_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_A_COLOUR_MODEL_LABELS_PATH = 'models/strawberry_colour_model_labels.txt'
STRAWBERRY_SUBCATEGORY_Strawberry_A_SIZE_MODEL_PATH = 'models/strawberry_size_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_A_SIZE_MODEL_LABELS_PATH = 'models/strawberry_size_model_labels.txt'
STRAWBERRY_SUBCATEGORY_Strawberry_A_SHAPE_MODEL_PATH = 'models/strawberry_shape_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_A_SHAPE_MODEL_LABELS_PATH = 'models/strawberry_shape_model_labels.txt'
STRAWBERRY_SUBCATEGORY_Strawberry_A_SURFACE_MODEL_PATH = 'models/strawberry_surface_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_A_SURFACE_MODEL_LABELS_PATH = 'models/strawberry_surface_model_labels.txt'

# Strawberry subcategory B
STRAWBERRY_SUBCATEGORY_Strawberry_B_COLOUR_MODEL_PATH = 'models/strawberry_colour_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_B_COLOUR_MODEL_LABELS_PATH = 'models/strawberry_colour_model_labels.txt'
STRAWBERRY_SUBCATEGORY_Strawberry_B_SIZE_MODEL_PATH = 'models/strawberry_size_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_B_SIZE_MODEL_LABELS_PATH = 'models/strawberry_size_model_labels.txt'
STRAWBERRY_SUBCATEGORY_Strawberry_B_SHAPE_MODEL_PATH = 'models/strawberry_shape_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_B_SHAPE_MODEL_LABELS_PATH = 'models/strawberry_shape_model_labels.txt'
STRAWBERRY_SUBCATEGORY_Strawberry_B_SURFACE_MODEL_PATH = 'models/strawberry_surface_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_B_SURFACE_MODEL_LABELS_PATH = 'models/strawberry_surface_model_labels.txt'

# Strawberry subcategory C
STRAWBERRY_SUBCATEGORY_Strawberry_C_COLOUR_MODEL_PATH = 'models/strawberry_colour_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_C_COLOUR_MODEL_LABELS_PATH = 'models/strawberry_colour_model_labels.txt'
STRAWBERRY_SUBCATEGORY_Strawberry_C_SIZE_MODEL_PATH = 'models/strawberry_size_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_C_SIZE_MODEL_LABELS_PATH = 'models/strawberry_size_model_labels.txt'
STRAWBERRY_SUBCATEGORY_Strawberry_C_SHAPE_MODEL_PATH = 'models/strawberry_shape_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_C_SHAPE_MODEL_LABELS_PATH = 'models/strawberry_shape_model_labels.txt'
STRAWBERRY_SUBCATEGORY_Strawberry_C_SURFACE_MODEL_PATH = 'models/strawberry_surface_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_C_SURFACE_MODEL_LABELS_PATH = 'models/strawberry_surface_model_labels.txt'

# Anthurium subcategory A
ANTHURIUM_SUBCATEGORY_Anthurium_A_HEALTHY_UNHEALTHY_MODEL_PATH = 'models/healthyAnthurium.h5'
ANTHURIUM_SUBCATEGORY_Anthurium_A_HEALTHY_UNHEALTHY_MODEL_LABELS_PATH = 'models/healthyAnthurium_labels.txt'
ANTHURIUM_SUBCATEGORY_Anthurium_A_COLOUR_MODEL_PATH = 'models/colourAnthurium.h5'
ANTHURIUM_SUBCATEGORY_Anthurium_A_COLOUR_MODEL_LABELS_PATH = 'models/colourAnthurium_labels.txt'
ANTHURIUM_SUBCATEGORY_Anthurium_A_SIZE_MODEL_PATH = 'models/sizeAnthurium.h5'
ANTHURIUM_SUBCATEGORY_Anthurium_A_SIZE_MODEL_LABELS_PATH = 'models/sizeAnthurium_labels.txt'

# Anthurium subcategory B
ANTHURIUM_SUBCATEGORY_Anthurium_B_HEALTHY_UNHEALTHY_MODEL_PATH = 'models/healthyAnthurium.h5'
ANTHURIUM_SUBCATEGORY_Anthurium_B_HEALTHY_UNHEALTHY_MODEL_LABELS_PATH = 'models/healthyAnthurium_labels.txt'
ANTHURIUM_SUBCATEGORY_Anthurium_B_COLOUR_MODEL_PATH = 'models/colourAnthurium.h5'
ANTHURIUM_SUBCATEGORY_Anthurium_B_COLOUR_MODEL_LABELS_PATH = 'models/colourAnthurium_labels.txt'
ANTHURIUM_SUBCATEGORY_Anthurium_B_SIZE_MODEL_PATH = 'models/sizeAnthurium.h5'
ANTHURIUM_SUBCATEGORY_Anthurium_B_SIZE_MODEL_LABELS_PATH = 'models/sizeAnthurium_labels.txt'

# Anthurium subcategory C
ANTHURIUM_SUBCATEGORY_Anthurium_C_HEALTHY_UNHEALTHY_MODEL_PATH = 'models/healthyAnthurium.h5'
ANTHURIUM_SUBCATEGORY_Anthurium_C_HEALTHY_UNHEALTHY_MODEL_LABELS_PATH = 'models/healthyAnthurium_labels.txt'
ANTHURIUM_SUBCATEGORY_Anthurium_C_COLOUR_MODEL_PATH = 'models/colourAnthurium.h5'
ANTHURIUM_SUBCATEGORY_Anthurium_C_COLOUR_MODEL_LABELS_PATH = 'models/colourAnthurium_labels.txt'
ANTHURIUM_SUBCATEGORY_Anthurium_C_SIZE_MODEL_PATH = 'models/sizeAnthurium.h5'
ANTHURIUM_SUBCATEGORY_Anthurium_C_SIZE_MODEL_LABELS_PATH = 'models/sizeAnthurium_labels.txt'

# BellPepper
BELL_PEPPER_DISEASE_MODEL_PATH = 'models/bellpepper_disease_model.h5'
BELL_PEPPER_DISEASE_MODEL_LABELS_PATH = 'models/bellpepper_disease_model_labels.txt'
BELL_PEPPER_HEALTHY_UNHEALTHY_MODEL_PATH = 'models/bellpepper_healthy_unhealthy_model.h5'
BELL_PEPPER_HEALTHY_UNHEALTHY_MODEL_LABELS_PATH = 'models/bellpepper_healthy_unhealthy_labels.txt'
BELL_PEPPER_MAGNESIUM_MODEL_PATH = 'models/bellpepper_initial_model.h5'
BELL_PEPPER_MAGNESIUM_MODEL_LABELS_PATH = 'models/bellpepper_initial_model_labels.txt'
BELL_PEPPER_POWDERY_MODEL_PATH = 'models/bellpepper_initial_severe_model.h5'
BELL_PEPPER_POWDERY_MODEL_LABELS_PATH = 'models/bellpepper_initial_severe_model_labels.txt'

rf = Roboflow(api_key="zpJpqip0iN5TpMWJPaM2")
project = rf.workspace().project("pepper-segmentation")
model = project.version(1).model


# Utility function to sanitize subcategory
def sanitize_subcategory(subcategory):
    return subcategory.replace(" ", "_")


# General prediction method
def getPrediction(filename, model_path, labels_path):
    try:
        np.set_printoptions(suppress=True)
        model = load_model(model_path, compile=False)
        class_names = open(labels_path, "r").readlines()

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image_path = 'uploads/' + filename
        openImage = Image.open(image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(openImage, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Convert numpy float32 to Python float
        confidence_score = round(float(confidence_score), 2)

        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)
        return class_name.strip(), confidence_score
    except Exception as e:
        print(f"Error in getPrediction: {e}")
        raise


# Anthurium methods


def getAnthuriumHealthPrediction(filename, subcategory):
    try:
        print(f"Received request for Anthurium Health Prediction for {subcategory}")
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'ANTHURIUM_SUBCATEGORY_{subcategory}_HEALTHY_UNHEALTHY_MODEL_PATH')
        labels_path = eval(f'ANTHURIUM_SUBCATEGORY_{subcategory}_HEALTHY_UNHEALTHY_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getAnthuriumClPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'ANTHURIUM_SUBCATEGORY_{subcategory}_COLOUR_MODEL_PATH')
        labels_path = eval(f'ANTHURIUM_SUBCATEGORY_{subcategory}_COLOUR_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getAnthuriumSiPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'ANTHURIUM_SUBCATEGORY_{subcategory}_SIZE_MODEL_PATH')
        labels_path = eval(f'ANTHURIUM_SUBCATEGORY_{subcategory}_SIZE_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


# Apple methods
def getAppleClPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'APPLE_SUBCATEGORY_{subcategory}_COLOUR_MODEL_PATH')
        labels_path = eval(f'APPLE_SUBCATEGORY_{subcategory}_COLOUR_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getAppleShPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'APPLE_SUBCATEGORY_{subcategory}_SHAPE_MODEL_PATH')
        labels_path = eval(f'APPLE_SUBCATEGORY_{subcategory}_SHAPE_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getAppleSiPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'APPLE_SUBCATEGORY_{subcategory}_SIZE_MODEL_PATH')
        labels_path = eval(f'APPLE_SUBCATEGORY_{subcategory}_SIZE_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


# Strawberry methods
def getStrawberryClPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'STRAWBERRY_SUBCATEGORY_{subcategory}_COLOUR_MODEL_PATH')
        labels_path = eval(f'STRAWBERRY_SUBCATEGORY_{subcategory}_COLOUR_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getStrawberryShPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'STRAWBERRY_SUBCATEGORY_{subcategory}_SHAPE_MODEL_PATH')
        labels_path = eval(f'STRAWBERRY_SUBCATEGORY_{subcategory}_SHAPE_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getStrawberrySiPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'STRAWBERRY_SUBCATEGORY_{subcategory}_SIZE_MODEL_PATH')
        labels_path = eval(f'STRAWBERRY_SUBCATEGORY_{subcategory}_SIZE_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getStrawberrySurPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'STRAWBERRY_SUBCATEGORY_{subcategory}_SURFACE_MODEL_PATH')
        labels_path = eval(f'STRAWBERRY_SUBCATEGORY_{subcategory}_SURFACE_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


# Mango methods
def getMangoClPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'MANGO_SUBCATEGORY_{subcategory}_COLOUR_MODEL_PATH')
        labels_path = eval(f'MANGO_SUBCATEGORY_{subcategory}_COLOUR_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getMangoShPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'MANGO_SUBCATEGORY_{subcategory}_SHAPE_MODEL_PATH')
        labels_path = eval(f'MANGO_SUBCATEGORY_{subcategory}_SHAPE_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getMangoSiPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'MANGO_SUBCATEGORY_{subcategory}_SIZE_MODEL_PATH')
        labels_path = eval(f'MANGO_SUBCATEGORY_{subcategory}_SIZE_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getMangoSurPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'MANGO_SUBCATEGORY_{subcategory}_SURFACE_MODEL_PATH')
        labels_path = eval(f'MANGO_SUBCATEGORY_{subcategory}_SURFACE_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


# BellPepper methods
def getBellPepperDisPrediction(filename):
    try:
        return getPrediction(filename, BELL_PEPPER_DISEASE_MODEL_PATH, BELL_PEPPER_DISEASE_MODEL_LABELS_PATH)
    except NameError:
        print("Bell Pepper disease model path not found.")
        raise


def getBellPepperHealthPrediction(filename):
    try:
        return getPrediction(filename, BELL_PEPPER_HEALTHY_UNHEALTHY_MODEL_PATH,
                             BELL_PEPPER_HEALTHY_UNHEALTHY_MODEL_LABELS_PATH)
    except NameError:
        print("Bell Pepper health model path not found.")
        raise


def getBellPepperMagPrediction(filename):
    try:
        return getPrediction(filename, BELL_PEPPER_MAGNESIUM_MODEL_PATH, BELL_PEPPER_MAGNESIUM_MODEL_LABELS_PATH)
    except NameError:
        print("Bell Pepper magnesium model path not found.")
        raise


def getBellPepperPowPrediction(filename):
    try:
        return getPrediction(filename, BELL_PEPPER_POWDERY_MODEL_PATH, BELL_PEPPER_POWDERY_MODEL_LABELS_PATH)
    except NameError:
        print("Bell Pepper powdery model path not found.")
        raise


# ExifData class and metadata extraction functions
class ExifData:
    def __init__(self, data):
        self.GPSInfo = data.get('GPSInfo', 'unknown')
        self.Make = data.get('Make', 'unknown')
        self.Model = data.get('Model', 'unknown')
        self.DateTime = data.get('DateTime', 'unknown')
        self.XResolution = data.get('XResolution', 'unknown')
        self.YResolution = data.get('YResolution', 'unknown')
        self.ExifVersion = data.get('ExifVersion', 'unknown')
        self.ApertureValue = data.get('ApertureValue', 'unknown')
        self.BrightnessValue = data.get('BrightnessValue', 'unknown')
        self.FocalLength = data.get('FocalLength', 'unknown')
        self.DigitalZoomRatio = data.get('DigitalZoomRatio', 'unknown')
        self.ExposureTime = data.get('ExposureTime', 'unknown')
        self.Contrast = data.get('Contrast', 'unknown')
        self.ISOSpeedRatings = data.get('ISOSpeedRatings', 'unknown')
        self.Saturation = data.get('Saturation', 'unknown')
        self.LensSpecification = data.get('LensSpecification', 'unknown')
        self.Sharpness = data.get('Sharpness', 'unknown')

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"


def get_image_metadata(image_path):
    img = Image.open(image_path)
    exif_data = img._getexif() or {}

    def get_tag_value(tag_id):
        return exif_data.get(tag_id, 'unknown')

    tags = {ExifTags.TAGS[k]: k for k in ExifTags.TAGS.keys()}
    extracted_data = {}
    for tag in tags:
        if tag in [
            'GPSInfo', 'Make', 'Model', 'DateTime', 'XResolution', 'YResolution',
            'ExifVersion', 'ApertureValue', 'BrightnessValue', 'FocalLength',
            'DigitalZoomRatio', 'ExposureTime', 'Contrast',
            'ISOSpeedRatings', 'Saturation', 'LensSpecification', 'Sharpness'
        ]:
            value = get_tag_value(tags[tag])
            if value is None:
                value = 'unknown'
            extracted_data[tag] = value

    if isinstance(extracted_data.get('GPSInfo', 'unknown'), dict):
        gps_info = extracted_data['GPSInfo']
        if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
            lat = gps_info['GPSLatitude']
            lon = gps_info['GPSLongitude']
            lat_ref = gps_info.get('GPSLatitudeRef', 'N')
            lon_ref = gps_info.get('GPSLongitudeRef', 'W')
            lat = (lat[0] + lat[1] / 60 + lat[2] / 3600) * (-1 if lat_ref == 'S' else 1)
            lon = (lon[0] + lon[1] / 60 + lon[2] / 3600) * (-1 if lon_ref == 'W' else 1)
        else:
            lat = None
            lon = None
    else:
        lat = None
        lon = None

    if isinstance(extracted_data.get('LensSpecification', 'unknown'), tuple):
        extracted_data['LensSpecification'] = tuple(
            x if x is not None else 'unknown' for x in extracted_data['LensSpecification'])
    else:
        extracted_data['LensSpecification'] = 'unknown'

    exif_record = ExifData(extracted_data)

    Date = extracted_data.get('DateTime', 'unknown')
    if Date != 'unknown':
        DateWork, Time = Date.split(' ')

    if lat is not None and lon is not None:
        geoLoc = Nominatim(user_agent="GetLoc")
        locname = geoLoc.reverse((lat, lon))
        Area = locname.address
    else:
        Area = "Unknown"

    attributes = [
        'Make', 'Model', 'XResolution', 'YResolution',
        'ExifVersion', 'ApertureValue', 'BrightnessValue', 'FocalLength',
        'DigitalZoomRatio', 'ExposureTime', 'Contrast',
        'ISOSpeedRatings', 'Saturation', 'LensSpecification', 'Sharpness'
    ]

    output_data = []
    output_data.append(f"Date: {DateWork.replace(':', '-')}")
    output_data.append(f"Time: {Time}")
    if lat is not None and lon is not None:
        output_data.append(f"Latitude: {lat:.6f}")
        output_data.append(f"Longitude: {lon:.6f}")
    else:
        output_data.append("Latitude: unknown")
        output_data.append("Longitude: unknown")
    output_data.append(f"Area: {Area}")

    for attr in attributes:
        value = getattr(exif_record, attr)
        if isinstance(value, tuple) or isinstance(value, dict):
            value = str(value)
        output_data.append(f"{attr}: {value}")

    return output_data


# Methods for Roboflow Predictions
def get_bellpepper_count(filename):
    try:
        image_path = 'uploads/' + filename
        response = model.predict(image_path, confidence=38, overlap=30).json()
        pepper_count = sum(1 for pred in response['predictions'] if pred['class'] == 'pepper')
        print("Number of Pepper Seeds \t" + str(pepper_count))
        return pepper_count
    except Exception as e:
        print(f"Error in get_pepper_count: {e}")
        raise


def get_papaw_count(filename):
    try:
        image_path = 'uploads/' + filename
        response = model.predict(image_path, confidence=38, overlap=30).json()
        papaw_count = sum(1 for pred in response['predictions'] if pred['class'] == 'papaw')
        print("Number of Papaw Seeds \t" + str(papaw_count))
        return papaw_count
    except Exception as e:
        print(f"Error in get_papaw_count: {e}")
        raise
