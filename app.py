from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os


from flask import make_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Use an absolute path for the "uploads" directory
UPLOAD_FOLDER = os.path.abspath('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def apply_blur(frame, face_coordinates):
    for (x, y, w, h) in face_coordinates:
        roi = frame[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (23, 23), 30)
        frame[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
    return frame

def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    face_coordinates = face_cascade.detectMultiScale(frame, 1.3, 5)
    return face_coordinates

def read_frame(file) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def process_video(file_path):
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        return None, {'error': 'Unable to open the video file'}

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_coordinates = detect_faces(frame)

        if len(face_coordinates) > 0:
            frame = apply_blur(frame, face_coordinates)
        frames.append(frame)

    cap.release()

    return frames, None

@app.route('/blur-faces', methods=['POST'])
def blur_faces():
    try:
        file = request.files['file']

        # Check if the file is a video
        _, file_extension = os.path.splitext(file.filename)
        if file_extension.lower() not in ['.mp4', '.avi', '.mkv']:
            return jsonify({'error': 'Invalid video format. Supported formats: mp4, avi, mkv'}), 400

        # Save the file to the absolute path
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)

        frames, error = process_video(video_path)

        if error:
            return jsonify(error), 500

        if frames:
            main_frame = frames[0]

            output_frames = []
            for frame in frames:
                image_pil = Image.fromarray(frame)
                img_byte_array = BytesIO()
                image_pil.save(img_byte_array, format="PNG")
                output_frames.append(img_byte_array.getvalue())

            response = make_response(send_file(BytesIO(output_frames[0]), mimetype='image/png', as_attachment=True,
                                              download_name='output_video_with_blur.mp4'))

            for idx, attachment in enumerate(output_frames[1:], 1):
                response.headers[f'Content-Disposition'] = f'attachment; filename=frame_{idx}.png'
                response.set_data(attachment)

            return response

        else:
            return jsonify({'error': 'No faces detected in the video'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
