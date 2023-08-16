from YoloClass import Model

model = Model()
model.load()

def get_yolo_predictions(img_array):
    return model.predict(img_array)



