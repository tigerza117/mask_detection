import signal
import sys
import threading
import time

from flask import Flask, render_template, request, copy_current_request_context, Response
import torch
import io
import base64
from PIL import Image
from flask_socketio import SocketIO, emit, disconnect
from threading import Lock
from flask_cors import CORS
import cv2

async_mode = None
model = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt", force_reload=True)
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'secret!'
socket_ = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()
buffered = io.BytesIO()
viewer = 0
camera = cv2.VideoCapture('intro.mp4')
running = True


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
        results = model(img, size=640)  # reduce size=320 for faster inference
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
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_byte = buffer.getvalue()
    encoded_img = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    return encoded_img


def gen_frames():
    while True:
        time.sleep(100 / 1000)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffered.getvalue() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    newViewer()
    time.sleep(1)
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def signal_handler(sig, frame):
    global running
    running = False
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def thread_callback():
    while running:
        if viewer < 1:
            time.sleep(0.5)
            continue
        success, frame = camera.read()  # read the camera frame
        if not success:
            camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            img = Image.open(io.BytesIO(frame))
            results = model(img, size=640)  # reduce size=320 for faster inference
            rendered_imgs = results.render()
            img_base64 = Image.fromarray(rendered_imgs[0])
            buffered.seek(0)
            buffered.truncate(0)
            img_base64.save(buffered, format="JPEG")
    sys.exit()


thr = threading.Thread(target=thread_callback)
thr.start()


@app.after_request
def response_processor(response):
    # Prepare all the local variables you need since the request context
    # will be gone in the callback function

    path = request.path

    @response.call_on_close
    def process_after_request():
        # Do whatever is necessary here
        if path == "/video_feed":
            closeViewer()
        pass

    return response


def newViewer():
    global viewer
    viewer += 1


def closeViewer():
    global viewer
    viewer -= 1


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False)
