from flask import Flask, render_template, Response
import cv2
import torch
import io
import base64
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt", force_reload=True)  # default
camera = cv2.VideoCapture(0)

def gen_frames():
    print(torch.cuda.is_available())
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            img = Image.open(io.BytesIO(frame))
            results = model(img, size=640)  # reduce size=320 for faster inference
            rendered_imgs = results.render()
            buffered = io.BytesIO()
            img_base64 = Image.fromarray(rendered_imgs[0])
            img_base64.save(buffered, format="JPEG")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffered.getvalue() + b'\r\n')


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False)
