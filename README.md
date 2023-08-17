# **QR Code reader using Yolov7**

## OBJECTIVE:
This is the repository for a basic localhost webapp that receives **.jpg, .jpeg or .png** files and outputs the **Yolov7 QR Code detection** on the image and tries to access the information inside this QR. It works under a **Flask App or Streamlit App**.

## EXAMPLE:


## USAGE:
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
### For Streamlit usage:
![image](https://github.com/joaosoutto95/yolov7_qrcode/assets/81658694/d51d5af3-b90a-45d1-a130-9faf6e6f5abd)
```
streamlit run .\streamlit_app.py
```
