from flask import Flask

# Define a flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




