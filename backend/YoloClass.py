# Imports:
import os
import sys
import torch
import numpy as np

from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device

# Input model parameters:
classes_to_filter = None  #You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]

opt  = {
    "weights": "./truora_model.pt", # Path to weights file default weights are for nano model
    "yaml"   : "data.yaml",
    "img-size": 640, # default image size
    "conf-thres": 0.5, # confidence threshold for inference.
    "iou-thres" : 0.2, # NMS IoU threshold for inference.
    "device" : 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes_to_filter,  # list of classes to filter or None
    "trace": True
}

# Model Class:

class Model():
    def __init__(self):
        self.loaded = False
        self.model_uri = opt['weights']
        self.model = None
        self.DEVICE = 'cpu'
        self.IMAGE_SIZE = opt['img-size']
    
    def load(self):
        # Initialize
        set_logging()
        device = select_device(self.DEVICE)
        self.half = device.type != 'cpu'  # half precision only supported on CUDA
   
        # check model exported from mlflow.tensorflow.autolog()
        self.model = attempt_load(self.model_uri, map_location=select_device(self.DEVICE))
        self.loaded = True
        if self.half:
            self.model.half()  # to FP16 

    def preprocess(self, img0):
        # Padded resize
        img = letterbox(img0)[0]
 
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return img

    
    def predict(self, X):
        try:
            # Revisar necessidade de load() toda vez
            if not self.loaded:
                self.load()
                
            image = X.astype(np.float32)
            
            # Preprocess the image
            image_pt = self.preprocess(image)
            image_pt = torch.from_numpy(image_pt).to(self.DEVICE)
            image_pt = image_pt.half() if self.half else image_pt.float()  # uint8 to fp16/32
            image_pt /= 255.0  # 0 - 255 to 0.0 - 1.0
            if image_pt.ndimension() == 3:
                image_pt = image_pt.unsqueeze(0)

            # Infer
            with torch.no_grad():
                pred = self.model(image_pt, augment=False)[0]

            # NMS
            pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'])
            # pred = non_max_suppression(pred)[0].cpu().numpy()
            
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # det = torch.tensor(det)
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(image_pt.shape[2:], det[:, :4], image.shape).round()

            result = {"detection_scores": [], "detection_boxes": [], 'detection_classes': []}
            
            for i in pred[0]:
                lsita_pred = i.tolist()
                result["detection_scores"].append(lsita_pred[4])
                result["detection_classes"].append(1)
                result["detection_boxes"].append([lsita_pred[1]/image.shape[0], 
                                                lsita_pred[0]/image.shape[1], 
                                                lsita_pred[3]/image.shape[0], 
                                                lsita_pred[2]/image.shape[1]
                                                ]
                    )
                
            return result
        
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)

            return {'1': exc_type, '2': fname, '3': exc_tb.tb_lineno, '4': e}

