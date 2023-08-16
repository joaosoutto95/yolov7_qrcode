import cv2
import numpy as np

from YoloClass import Model

def read_qr_code(img_array_original):
    try:
        model = Model()
        results = model.predict(img_array_original)
        qrc_results_arrays = []
        for result in results['detection_boxes']:
            y1, x1, y2, x2 = result
            height, width, _ = img_array_original.shape
            
            base_box = abs(x2 - x1)
            height_box = abs(y2 - y1)

            x1_pixel = max(int((x1-base_box*0.05) * width), 0)
            y1_pixel = max(int((y1-height_box*0.05) * height), 0)
            x2_pixel = min(int((x2+base_box*0.05) * width), width)
            y2_pixel = min(int((y2+height_box*0.05) * height), height)

            mask = np.zeros_like(img_array_original)
            mask[y1_pixel:y2_pixel, x1_pixel:x2_pixel] = img_array_original[y1_pixel:y2_pixel, x1_pixel:x2_pixel]
            qrc_results_arrays.append(mask)
        
        qrc_results = []
        for qrc in qrc_results_arrays:
            detect = cv2.QRCodeDetector()
            value, points, straight_qrcode = detect.detectAndDecode(qrc)
            if value == '':
                qrc_results.append('No results for this QR Code')
            else:
                qrc_results.append(value)
 
        return results['detection_boxes'], qrc_results
    except:
        return print('Not detected - Try a better image angle or quality.')
