import streamlit as st
from detect_qrcode import *
import cv2

st.title("QR Code Detection App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    results, qrc_results = read_qr_code(image)
    
    img_array = image.copy()
    height, width, _ = img_array.shape

    for result in results:
        y1, x1, y2, x2 = result

        base_box = abs(x2 - x1)
        height_box = abs(y2 - y1)

        x1_pixel = max(int((x1 - base_box * 0.05) * width), 0)
        y1_pixel = max(int((y1 - height_box * 0.05) * height), 0)
        x2_pixel = min(int((x2 + base_box * 0.05) * width), width)
        y2_pixel = min(int((y2 + height_box * 0.05) * height), height)

        pt1 = int(x1_pixel), int(y1_pixel)
        pt2 = int(x2_pixel), int(y2_pixel)

        cv2.rectangle(img_array, pt1, pt2, (0, 0, 255), 2)
        cv2.putText(img_array, f'QR CODE {results.index(result)+1}', (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    st.image(cv2.resize(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), (int(width*0.5), int(height*0.5))), caption="Processed Image")

    qrc_results_str = "\n".join(qrc_results) if isinstance(qrc_results, list) else qrc_results
    st.markdown("### QR Code Extraction Results")
    st.text_area("Results", qrc_results_str)


