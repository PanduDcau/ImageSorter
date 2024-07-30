import os
import flask
import json
from flask import render_template, request, send_file
from flask import jsonify
from werkzeug.utils import secure_filename
from app import app
from main import getBellpepperPrediction, getHealthyOrNotPrediction, getMagnesiumSeverePrediction, \
    getPowderyMildewPrediction, getPowderyOrMagnesiumPrediction


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predictBellpepper', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'})
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            label, acc = getBellpepperPrediction(filename)
            response = {
                'label': label,
                'probability': acc
            }
            return jsonify(response)
    else:
        return jsonify({'error': 'Invalid request method'})


@app.route('/predictHealthyOrNot', methods=['POST'])
def submit_file_color():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'})
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            label, acc = getHealthyOrNotPrediction(filename)
            response = {
                'label': label,
                'probability': acc
            }
            return jsonify(response)
    else:
        return jsonify({'error': 'Invalid request method'})


@app.route('/predictMagnesiumSevere', methods=['POST'])
def submit_file_leafMag():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'})
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            label, acc, infectedArea = getMagnesiumSeverePrediction(filename)
            response = {
                'label': label,
                'probability': acc,
                'infectedArea': infectedArea
            }
            return jsonify(response)
    else:
        return jsonify({'error': 'Invalid request method'})


@app.route('/predictPowderyOrMagnesium', methods=['POST'])
def submit_file_leafPowMag():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'})
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            label, acc = getPowderyOrMagnesiumPrediction(filename)
            response = {
                'label': label,
                'probability': acc
            }
            return jsonify(response)
    else:
        return jsonify({'error': 'Invalid request method'})


@app.route('/predictPowderyMildew', methods=['POST'])
def submit_file_leafPow():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'})
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            label, acc, infectedArea = getPowderyMildewPrediction(filename)
            response = {
                'label': label,
                'probability': acc,
                'infectedArea': infectedArea
            }
            return jsonify(response)
    else:
        return jsonify({'error': 'Invalid request method'})


@app.route('/getSeverityImage', methods=['GET'])
def get_image():
    image_path = 'output_images/color_patch_2.png'
    return send_file(image_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='192.168.1.2', port=4000)
