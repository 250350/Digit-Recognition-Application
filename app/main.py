"""Load the saved model once and handle drawing requests."""
from pathlib import Path

from flask import Flask, render_template, request
from tensorflow import keras
from werkzeug.exceptions import RequestEntityTooLarge

from app.preprocessing import InvalidDrawing, prepare_drawing

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
model_path = Path(__file__).resolve().parent / 'models/mnist_ann.keras'
model = keras.models.load_model(model_path, compile=False)


@app.route('/', methods=['GET', 'POST'])
def drawing():
    if request.method == 'GET':
        return render_template('drawing.html')

    data = request.form.get('canvasimg', '')
    try:
        image = prepare_drawing(data)
    except InvalidDrawing as exc:
        return render_template('drawing.html', error=str(exc)), 400

    try:
        scores = model(image, training=False).numpy()[0]
        return render_template('drawing.html', scores=scores.tolist(),
                               best=int(scores.argmax()), canvasdata=data)
    except Exception:
        app.logger.exception('Prediction failed')
        return render_template('drawing.html', error='Recognition failed. Please try again.'), 500


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return render_template('drawing.html', error='The image is too large (maximum 2 MB).'), 413
