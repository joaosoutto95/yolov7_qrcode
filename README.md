# **QR Code reader using Yolov7**

## OBJECTIVE:
This is the repository for a basic localhost webapp that receives **.jpg, .jpeg or .png** files and outputs the **Yolov7 QR Code detection** on the image and tries to access the information inside this QR. It works under a **Flask App or Streamlit App**.

## DATASET:
I've used the RoboFlow QR Code dataset available at the link bellow to train my algorithm (did a transfer learning from yolov7.pt, the COCO pre-trained weights):
https://universe.roboflow.com/qrcodes/qrcodes-same-split-oneclass/dataset/3

## REASONING BEHIND THE SOLUTION:
Basically i'm using a Yolov7 detection algorithm that receives and image array using cv2 to identify all QR Codes and list them so it can run one by one in a cv2 function named QRCodeDetector. This outputs the content, if possible, inside the QR Code.
After the model backend finishes it's work, the app.py or the streamlit_app.py organizes the output in a html page under a localhost server using Flask or Streamlit.

## INSTRUCTIONS FOR THE USAGE:
For starters, this webapp ran in a **Python 3.8.9** enviroment. So maybe check for dependency conflicts if your using another python version.
1. First you should create a folder in which the project will be git cloned, then use the code on Windows PS (or another CMD) bellow to clone the repository to your folder:
```
git clone https://github.com/joaosoutto95/yolov7_qrcode.git
```

2. Setup a virtual enviroment for the isolation of the project dependencies and then activate it:
In Windows PS the code used is:
```
python -m venv env
```
Activate:
```
.\env\Scripts\activate
```

3. After setting the venv, via PS enter the backend folder and:
```
pip install -r requirements.txt
```

4. Finally you can choose your webframework for deploying the webapp in the your localhost, the images bellow are followed by the code used: 
### For Flask usage:
![image](https://github.com/joaosoutto95/yolov7_qrcode/assets/81658694/b552a48c-3663-4d03-bc08-4cc3f367aac4)
```
python .\app.py
```
Now click on "Choose image" button and you should see the output of the image with the detection boxes and the QR information results bellow:
![image](https://github.com/joaosoutto95/yolov7_qrcode/assets/81658694/e19c87fa-c1f9-4c50-a9a7-7e26e27f5505)


### For Streamlit usage:
![image](https://github.com/joaosoutto95/yolov7_qrcode/assets/81658694/d51d5af3-b90a-45d1-a130-9faf6e6f5abd)
```
streamlit run .\streamlit_app.py
```
Now drop or browse an QR Code image and you should see the output of the image with the detection boxes and the QR informatio results bellow:
![image](https://github.com/joaosoutto95/yolov7_qrcode/assets/81658694/5d2f5884-6b4d-4e25-a155-c3b90248afb5)


## BONUS:
In this example, the QR Code is rotated and the Yolov7 detects while the cv2 predicts correctly it's content.
![image](https://github.com/joaosoutto95/yolov7_qrcode/assets/81658694/0a8b1b79-65df-4415-bdac-c50ef20a22c7)
