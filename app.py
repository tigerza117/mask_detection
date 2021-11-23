from flask import Flask, render_template, request, send_file, session, copy_current_request_context
import torch
import io
import base64
from PIL import Image
from flask_socketio import SocketIO, emit, disconnect
from threading import Lock
from flask_cors import CORS

async_mode = None
model = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt", force_reload=True)
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'secret!'
socket_ = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()


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
        # buffered = io.BytesIO()
        img_base64 = Image.fromarray(results.render()[0])
        # img_base64.save(buffered, format="JPEG")
        # buffered.seek(0)
        return base64_encode_img(img_base64)
        # return send_file(buffered, mimetype='image/jpeg')


@socket_.on('process', namespace='/predict')
def ws_predict(message):
    # Convert to PIL image
    data = message['data']
    image = data[data.find(",") + 1:]
    dec = base64.b64decode(image + "===")
    img = Image.open(io.BytesIO(dec)).convert("RGB")

    # Process the image
    results = model(img, size=320)  # reduce size=320 for faster inference
    img_base64 = Image.fromarray(results.render()[0])

    emit('process_response',
         {'data': base64_encode_img(img_base64)})


@socket_.on('disconnect_request', namespace='/predict')
def disconnect_request():
    @copy_current_request_context
    def can_disconnect():
        disconnect()


def base64_encode_img(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    encoded_img = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    return encoded_img


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False)
