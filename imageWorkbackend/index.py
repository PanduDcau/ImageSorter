import os
from flask import render_template, request, jsonify
from werkzeug.utils import secure_filename
from app import app
from main import getAppleClPrediction, getAppleShPrediction, getAppleSiPrediction, getStrawberryClPrediction, \
    getStrawberryShPrediction, getStrawberrySiPrediction, getStrawberrySurPrediction, \
    getMangoClPrediction, getMangoShPrediction, getMangoSiPrediction, getMangoSurPrediction, \
    getBellPepperDisPrediction, getBellPepperHealthPrediction, getBellPepperMagPrediction, getBellPepperPowPrediction, \
    get_image_metadata, get_papaw_count, get_bellpepper_count

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


def handle_file_upload(request):
    if 'file' not in request.files:
        return None, {'error': 'No file part'}
    file = request.files['file']
    if file.filename == '':
        return None, {'error': 'No file selected for uploading'}
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return filename, None
    return None, {'error': 'Invalid request method'}


@app.route('/predictAppleCl/<subcategory>', methods=['POST'])
def submit_acl(subcategory):
    filename, error = handle_file_upload(request)
    if error:
        return jsonify(error)
    label, acc = getAppleClPrediction(filename, subcategory)
    return jsonify({'label': label, 'probability': acc})


@app.route('/predictAppleSh/<subcategory>', methods=['POST'])
def submit_ash(subcategory):
    filename, error = handle_file_upload(request)
    if error:
        return jsonify(error)
    label, acc = getAppleShPrediction(filename, subcategory)
    return jsonify({'label': label, 'probability': acc})


@app.route('/predictAppleSi/<subcategory>', methods=['POST'])
def submit_asi(subcategory):
    filename, error = handle_file_upload(request)
    if error:
        return jsonify(error)
    label, acc = getAppleSiPrediction(filename, subcategory)
    return jsonify({'label': label, 'probability': acc})


@app.route('/predictStrawberryCl/<subcategory>', methods=['POST'])
def submit_scl(subcategory):
    filename, error = handle_file_upload(request)
    if error:
        return jsonify(error)
    label, acc = getStrawberryClPrediction(filename, subcategory)
    return jsonify({'label': label, 'probability': acc})


@app.route('/predictStrawberrySh/<subcategory>', methods=['POST'])
def submit_ssh(subcategory):
    filename, error = handle_file_upload(request)
    if error:
        return jsonify(error)
    label, acc = getStrawberryShPrediction(filename, subcategory)
    return jsonify({'label': label, 'probability': acc})


@app.route('/predictStrawberrySi/<subcategory>', methods=['POST'])
def submit_ssi(subcategory):
    filename, error = handle_file_upload(request)
    if error:
        return jsonify(error)
    label, acc = getStrawberrySiPrediction(filename, subcategory)
    return jsonify({'label': label, 'probability': acc})


@app.route('/predictStrawberrySur/<subcategory>', methods=['POST'])
def submit_ssur(subcategory):
    filename, error = handle_file_upload(request)
    if error:
        return jsonify(error)
    label, acc = getStrawberrySurPrediction(filename, subcategory)
    return jsonify({'label': label, 'probability': acc})


@app.route('/predictMangoCl/<subcategory>', methods=['POST'])
def submit_mcl(subcategory):
    filename, error = handle_file_upload(request)
    if error:
        return jsonify(error)
    label, acc = getMangoClPrediction(filename, subcategory)
    return jsonify({'label': label, 'probability': acc})


@app.route('/predictMangoSh/<subcategory>', methods=['POST'])
def submit_msh(subcategory):
    filename, error = handle_file_upload(request)
    if error:
        return jsonify(error)
    label, acc = getMangoShPrediction(filename, subcategory)
    return jsonify({'label': label, 'probability': acc})


@app.route('/predictMangoSi/<subcategory>', methods=['POST'])
def submit_msi(subcategory):
    filename, error = handle_file_upload(request)
    if error:
        return jsonify(error)
    label, acc = getMangoSiPrediction(filename, subcategory)
    return jsonify({'label': label, 'probability': acc})


@app.route('/predictMangoSur/<subcategory>', methods=['POST'])
def submit_msur(subcategory):
    filename, error = handle_file_upload(request)
    if error:
        return jsonify(error)
    label, acc = getMangoSurPrediction(filename, subcategory)
    return jsonify({'label': label, 'probability': acc})


@app.route('/predictBellPepperDis', methods=['POST'])
def submit_dis():
    filename, error = handle_file_upload(request)
    if error:
        return jsonify(error)
    label, acc = getBellPepperDisPrediction(filename)
    return jsonify({'label': label, 'probability': acc})


@app.route('/predictBellPepperHealth', methods=['POST'])
def submit_health():
    filename, error = handle_file_upload(request)
    if error:
        return jsonify(error)
    label, acc = getBellPepperHealthPrediction(filename)
    return jsonify({'label': label, 'probability': acc})


@app.route('/predictBellPepperMag', methods=['POST'])
def submit_mag():
    filename, error = handle_file_upload(request)
    if error:
        return jsonify(error)
    label, acc = getBellPepperMagPrediction(filename)
    return jsonify({'label': label, 'probability': acc})


@app.route('/predictBellPepperPow', methods=['POST'])
def submit_pow():
    filename, error = handle_file_upload(request)
    if error:
        return jsonify(error)
    label, acc = getBellPepperPowPrediction(filename)
    return jsonify({'label': label, 'probability': acc})


@app.route('/predictPepperCount', methods=['POST'])
def submit_pepper_count():
    filename, error = handle_file_upload(request)
    if error:
        return jsonify(error)
    count = get_bellpepper_count(filename)
    return jsonify({'count': count})


@app.route('/predictPapawCount', methods=['POST'])
def submit_papaw_count():
    filename, error = handle_file_upload(request)
    if error:
        return jsonify(error)
    count = get_papaw_count(filename)
    return jsonify({'count': count})


@app.route('/predictEXIF', methods=['POST'])
def submit_exif():
    filename, error = handle_file_upload(request)
    if error:
        return jsonify(error)
    metadata = get_image_metadata('uploads/' + filename)
    return jsonify({'metadata': metadata})


if __name__ == '__main__':
    app.run(debug=True, host='192.168.84.24', port=4000)
