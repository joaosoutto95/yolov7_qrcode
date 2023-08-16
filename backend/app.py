from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from detect_qrcode import *

import json
import cv2
import os
import base64

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            results, qrc_results = read_qr_code(cv2.imread(filepath))
            
            img_array = cv2.imread(filepath)
            height, width, _ = img_array.shape

            for result in results:
                y1, x1, y2, x2 = result

                base_box = abs(x2 - x1)
                height_box = abs(y2 - y1)

                x1_pixel = max(int((x1-base_box*0.05) * width), 0)
                y1_pixel = max(int((y1-height_box*0.05) * height), 0)
                x2_pixel = min(int((x2+base_box*0.05) * width), width)
                y2_pixel = min(int((y2+height_box*0.05) * height), height)

                pt1 = int(x1_pixel), int(y1_pixel)
                pt2 = int(x2_pixel), int(y2_pixel)

                cv2.rectangle(img_array, pt1, pt2, (0, 0, 255), 2)  # Red color
                cv2.putText(img_array, f'QR CODE {results.index(result)+1}', (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imwrite(filepath, cv2.resize(img_array, (int(width*0.5), int(height*0.5))))

            return redirect(url_for('show_result', filepath=filepath, qrc_results=json.dumps(qrc_results)))

    return render_template('index.html')

@app.route('/show_result', methods=['GET', 'POST'])
def show_result():
    if request.method == 'GET':
        filepath = request.args.get('filepath')
        qrc_results = json.loads(request.args.get('qrc_results'))

        image_base64 = None
        with open(filepath, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        if isinstance(qrc_results, str):
            qrc_results = [(1, qrc_results)]

        return render_template('result.html', image_url=f'data:image/jpeg;base64,{image_base64}', qr_codes=qrc_results)
    
    return "Please upload an image first."

if __name__ == '__main__':
    app.run(debug=True)