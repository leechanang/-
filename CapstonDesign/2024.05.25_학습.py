import multiprocessing
from ultralytics import YOLO

if __name__ == '__main__':
    multiprocessing.freeze_support() #windows에서는 필수
    #Load a model
    #model = YOLO("yolov8n.yaml") # build a new model from scratch
    model = YOLO(model='yolov8n.pt', task='detect')  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data=r"C:\\Users\\Chan's Victus\\Desktop\\pythonProject\\capstonD\\data\\dataset\\data.yaml",
                epochs = 30,
                batch = 10,
                imgsz = 640,
                ) # train the model

    model.val() # evaluate model performance on the validation set
    model(r"C:\Users\Chan's Victus\Desktop\pythonProject\capstonD\data\dataset\images\train\MVI_0788_VIS_frame0.jpg") #predict on an image
    success = model.export(format='onnx') # export the model to ONNX format

