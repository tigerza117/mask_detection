from flask import Flask, render_template, request, send_file
import cv2
import torch
import io
import base64
from PIL import Image


model = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt", force_reload=True)
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()

        img = Image.open(io.BytesIO(image_bytes))

        results = model(img, size=320)  # reduce size=320 for faster inference
        buffered = io.BytesIO()
        img_base64 = Image.fromarray(results.render()[0])
        img_base64.save(buffered, format="JPEG")
        buffered.seek(0)
        return send_file(buffered, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False)
