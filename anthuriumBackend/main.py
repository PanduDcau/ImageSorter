from teachable_machine import TeachableMachine
import os
import cv2
import numpy as np
import requests
from PIL import Image
from rembg import remove

# Model Saved Path
# DETECT_BELLPEPPER_MODEL_PATH = 'models/detect_bellpepper.h5'
# DETECT_BELLPEPPER_MODEL_LABELS_PATH = 'models/detect_bellpepper_labels.txt'

DETECT_BELLPEPPER_MODEL_PATH = 'models/healthyAnthurium.h5'
DETECT_BELLPEPPER_MODEL_LABELS_PATH = 'models/healthyAnthurium_labels.txt'

# HEALTHY_OR_NON_HEALTHY_MODEL_PATH = 'models/healthy_nonhealthy_model.h5'
# HEALTHY_OR_NON_HEALTHY_MODEL_LABELS_PATH = 'models/healthy_nonhealthy_model_labels.txt'

HEALTHY_OR_NON_HEALTHY_MODEL_PATH = 'models/sizeAnthurium.h5'
HEALTHY_OR_NON_HEALTHY_MODEL_LABELS_PATH = 'models/sizeAnthurium_labels.txt'

# POWDERY_OR_MAGNESIUM_MODEL_PATH = 'models/powdery_or_magnesium_model.h5'
# POWDERY_OR_MAGNESIUM_MODEL_LABELS_PATH = 'models/powdery_or_magnesium_model_labels.txt'

POWDERY_OR_MAGNESIUM_MODEL_PATH = 'models/colourAnthurium.h5'
POWDERY_OR_MAGNESIUM_MODEL_LABELS_PATH = 'models/colourAnthurium_labels.txt'

MAGNESIUM_SEVERE_MODEL_PATH = 'models/magnesium_severe_model.h5'
MAGNESIUM_SEVERE_MODEL_LABELS_PATH = 'models/magnesium_severe_model_labels.txt'

POWDERYMILDEW_MODEL_PATH = 'models/powderymildew_model.h5'
POWDERYMILDEW_MODEL_LABELS_PATH = 'models/powderymildew_model_labels.txt'

# Define the mapping from class indices to class labels
# detect_bellpepper_labels = {0: 'bell paper', 1: 'not bell paper'}
# healthy_non_healthy_labels = {0: 'healthy', 1: 'non healthy'}
# magnesium_severe_labels = {0: 'Initial Stage', 1: 'Severity Stage'}
# powderymildew_labels = {0: 'Initial Stage', 1: 'Severity Stage'}
# powdery_or_magnesium_labels = {0: 'Powdery Mildew', 1: 'Magnesium Severe'}

detect_bellpepper_labels = {0: 'Healthy Anthurium', 1: 'Not Healthy Anthurium'}
# healthy_non_healthy_labels = {0: 'Healthy Bell Pepper', 1: 'Not Healthy Bell Pepper'}
healthy_non_healthy_labels = {0: 'Small flower', 1: 'Large flower'}
powdery_or_magnesium_labels = {0: 'Red', 1: 'White'}
magnesium_severe_labels = {0: 'Initial Stage', 1: 'Severity Stage'}
powderymildew_labels = {0: 'Initial Stage', 1: 'Severity Stage'}

def getBellpepperPrediction(filename):
    model = TeachableMachine(model_path=DETECT_BELLPEPPER_MODEL_PATH,
                             labels_file_path=DETECT_BELLPEPPER_MODEL_LABELS_PATH)

    image_path = 'uploads/' + filename

    result = model.classify_image(image_path)

    print("class_index", result["class_index"])
    print("class_name:::", result["class_name"])
    print("class_confidence:", result["class_confidence"])

    label = str(result["class_name"])
    probability = str(result["class_confidence"])
    return label, probability


def getHealthyOrNotPrediction(filename):
    model = TeachableMachine(model_path=HEALTHY_OR_NON_HEALTHY_MODEL_PATH,
                             labels_file_path=HEALTHY_OR_NON_HEALTHY_MODEL_LABELS_PATH)

    image_path = 'uploads/' + filename

    result = model.classify_image(image_path)

    print("class_index", result["class_index"])

    print("class_name:::", result["class_name"])

    print("class_confidence:", result["class_confidence"])

    colorLabel = str(result["class_name"])
    probability = str(result["class_confidence"])

    return colorLabel, probability


def getMagnesiumSeverePrediction(filename):
    model = TeachableMachine(model_path=MAGNESIUM_SEVERE_MODEL_PATH,
                             labels_file_path=MAGNESIUM_SEVERE_MODEL_LABELS_PATH)

    image_path = 'uploads/' + filename

    result = model.classify_image(image_path)

    print("class_index", result["class_index"])

    print("class_name:::", result["class_name"])
    print(f"Length of class_name: {len(result['class_name'])}")

    print("class_confidence:", result["class_confidence"])

    label = str(result["class_name"])
    probability = str(result["class_confidence"])

    original_input_image = 'uploads/' + filename
    input_image = 'uploads/' + filename.split('.')[0] + '.png'
    output_image_folder = 'output_images'
    os.makedirs(output_image_folder, exist_ok=True)

    remove_background(original_input_image, input_image)
    areas, centers = segment_color_patches(input_image, output_image_folder)
    nearest_green_idx, nearest_green_center = find_nearest_to_green(centers)
    nearest_yellow_idx, nearest_yellow_center = find_nearest_to_yellow(centers)
    total_area = sum(areas)
    # area_percentages = [area / areas[nearest_green_idx] * 100 for area in areas]
    area_percentages = [area / total_area * 100 for area in areas]

    for i, (area, center, percentage) in enumerate(zip(areas, centers, area_percentages)):
        print(
            f"Area of color patch {i + 1}: {area} pixels ({percentage:.2f}%) - Color code: {tuple(center.astype(int))}")
        if i == 2:
            class_name = result["class_name"].strip()
            if class_name == "Initial Stage":
                print("came inside 1")
                infected_area_percentage = percentage
                if infected_area_percentage >= 50:
                    infected_area_percentage = infected_area_percentage - 10
            else:
                print("came inside 2")
                infected_area_percentage = percentage
                infected_area_percentage = 100 - infected_area_percentage

        if i == nearest_green_idx:
            print("This color patch is the nearest to green.")
        if i == nearest_yellow_idx:
            print("This color patch is the nearest to yellow.")

    # infected_area_percentage = (areas[nearest_yellow_idx] / areas[nearest_green_idx]) * 100
    # green_percentage = areas[nearest_green_idx]
    # print(f"Green area percentage = {green_percentage:.2f}")
    # infected_area_percentage = 100 - areas[nearest_green_idx]
    print(f"Infected area percentage = {infected_area_percentage:.2f}")

    return label, probability, round(infected_area_percentage, 2)


def getPowderyMildewPrediction(filename):
    model = TeachableMachine(model_path=POWDERYMILDEW_MODEL_PATH,
                             labels_file_path=POWDERYMILDEW_MODEL_LABELS_PATH)

    image_path = 'uploads/' + filename

    result = model.classify_image(image_path)

    print("class_index", result["class_index"])

    print("class_name:::", result["class_name"])

    print("class_confidence:", result["class_confidence"])

    label = str(result["class_name"])
    probability = str(result["class_confidence"])

    original_input_image = 'uploads/' + filename
    input_image = 'uploads/' + filename.split('.')[0] + '.png'
    output_image_folder = 'output_images'
    os.makedirs(output_image_folder, exist_ok=True)

    remove_background(original_input_image, input_image)
    areas, centers = segment_color_patches(input_image, output_image_folder)
    nearest_green_idx, nearest_green_center = find_nearest_to_green(centers)
    nearest_yellow_idx, nearest_yellow_center = find_nearest_to_yellow(centers)
    total_area = sum(areas)
    # area_percentages = [area / areas[nearest_green_idx] * 100 for area in areas]
    area_percentages = [area / total_area * 100 for area in areas]

    for i, (area, center, percentage) in enumerate(zip(areas, centers, area_percentages)):
        print(
            f"Area of color patch {i + 1}: {area} pixels ({percentage:.2f}%) - Color code: {tuple(center.astype(int))}")
        if i == 2:
            class_name = result["class_name"].strip()
            if class_name == "Initial Stage":
                print("came inside 1")
                infected_area_percentage = percentage
                if infected_area_percentage >= 50:
                    infected_area_percentage = infected_area_percentage - 10
            else:
                print("came inside 2")
                infected_area_percentage = percentage
                infected_area_percentage = 100 - infected_area_percentage
        if i == nearest_green_idx:
            print("This color patch is the nearest to green.")
        if i == nearest_yellow_idx:
            print("This color patch is the nearest to yellow.")

    # infected_area_percentage = (areas[nearest_yellow_idx] / areas[nearest_green_idx]) * 100
    # green_percentage = areas[nearest_green_idx]
    # print(f"Green area percentage = {green_percentage:.2f}")
    # infected_area_percentage = 100 - areas[nearest_green_idx]
    print(f"Infected area percentage = {infected_area_percentage:.2f}")

    return label, probability, round(infected_area_percentage, 2)


def getPowderyOrMagnesiumPrediction(filename):
    model = TeachableMachine(model_path=POWDERY_OR_MAGNESIUM_MODEL_PATH,
                             labels_file_path=POWDERY_OR_MAGNESIUM_MODEL_LABELS_PATH)
    image_path = 'uploads/' + filename
    result = model.classify_image(image_path)
    print("class_index", result["class_index"])

    print("class_name:::", result["class_name"])

    print("class_confidence:", result["class_confidence"])

    label = str(result["class_name"])
    probability = str(result["class_confidence"])

    return label, probability


def remove_background(input_image_path, output_image_path):
    # Check if u2net.onnx exists in the models directory
    models_directory = 'models'
    model_file_path = os.path.join(models_directory, 'u2net.onnx')
    if not os.path.exists(model_file_path):
        # Download u2net.onnx if it does not exist
        os.makedirs(models_directory, exist_ok=True)
        download_url = 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx'
        download_file(download_url, model_file_path)
    # Remove the background
    with open(input_image_path, "rb") as img_file:
        input_image = Image.open(img_file)
        output_image = remove(input_image, model_file=model_file_path)
        output_image.save(output_image_path)


def download_file(url, file_path):
    # Download file from url and save to file_path
    with open(file_path, 'wb') as f:
        response = requests.get(url)
        f.write(response.content)


def euclidean_distance(color1, color2):
    return np.linalg.norm(np.array(color1) - np.array(color2))


def find_nearest_to_green(centers):
    green = (0, 255, 0)
    distances = [euclidean_distance(center, green) for center in centers]
    nearest_green_idx = np.argmin(distances)
    return nearest_green_idx, centers[nearest_green_idx]


def find_nearest_to_yellow(centers):
    yellow = (99, 126, 56)
    distances = [euclidean_distance(center, yellow) for center in centers]
    nearest_yellow_idx = np.argmin(distances)
    return nearest_yellow_idx, centers[nearest_yellow_idx]


def segment_color_patches(input_image_path, output_folder, num_colors=3):
    input_image = Image.open(input_image_path)
    instance_image = np.asarray(input_image)

    if instance_image.shape[2] == 4:  # If input image has an alpha channel
        instance_image = instance_image[:, :, :3]  # Remove the alpha channel

    # Prepare data for k-means clustering
    data = instance_image.reshape((-1, 3))
    valid_pixels = data.sum(axis=1) != 0  # Identify background pixels
    data = data[valid_pixels]  # Remove background pixels
    data = np.float32(data)

    # Perform k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(data, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    areas = [0] * num_colors

    for i, center in enumerate(centers):
        output_image = np.zeros_like(instance_image)

        for y in range(instance_image.shape[0]):
            for x in range(instance_image.shape[1]):
                if not np.all(instance_image[y, x] == 0):  # If the pixel is not background
                    distances = [np.linalg.norm(instance_image[y, x] - c) for c in centers]
                    nearest_center_idx = np.argmin(distances)

                    if nearest_center_idx == i:
                        output_image[y, x] = center
                        areas[i] += 1

        output_image_path = os.path.join(output_folder, f"color_patch_{i + 1}.png")
        cv2.imwrite(output_image_path, output_image)
        print(f"Color patch {i + 1} segmented in '{input_image_path}' and saved to '{output_image_path}'.")
    return areas, centers
