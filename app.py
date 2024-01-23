from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


from flask_cors import CORS
CORS(app) 


app = Flask(__name__)

def apply_blur(image, face_coordinates):
    for (x, y, w, h) in face_coordinates:
        roi = image[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (23, 23), 30)
        image[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
    return image

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    face_coordinates = face_cascade.detectMultiScale(image, 1.3, 5)
    return face_coordinates

def read_imagefile(file) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

@app.route('/blur-faces', methods=['POST'])
def blur_faces():
    try:
        file = request.files['file']
        image = read_imagefile(file)
        face_coordinates = detect_faces(image)
        if len(face_coordinates) > 0:
            image = apply_blur(image, face_coordinates)
            image_pil = Image.fromarray(image)
            img_byte_array = BytesIO()
            image_pil.save(img_byte_array, format="PNG")
            return send_file(BytesIO(img_byte_array.getvalue()), mimetype='image/png')
        else:
            return jsonify({'error': 'No faces detected in the image'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')


# if __name__ == '__main__':
#     app.run(port=5000, debug=True)



# if __name__ == 'app.py':
#     app.run(debug=True)

# http://127.0.0.1:5000/blur-faces