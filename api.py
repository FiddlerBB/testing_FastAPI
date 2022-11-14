
"""
How to use:

    -> Run for demo FastAPI backend api just follow command line: uvicorn api:app

"""

import io
from PIL import Image
from fastapi import File, FastAPI
from text_recognition.vietocr.tool.predictor import Predictor
from text_recognition.vietocr.tool.config import Cfg
import torch

"""
This script is use to config Text recognition model and then load model preparing for prediction 
"""
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'model/transformerocr.pth'
config['cnn']['pretrained'] = False
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False
detector = Predictor(config)

# Load Text Detection Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/text_detection.pt')

# create FastAPI
app = FastAPI()


# integrate Model to API using POST method
@app.post("/scan/text-detection/")
async def get_body(file: bytes = File(...)):
    """
    -> We have  + confidence of text -> result[4]
             + xmin,ymin,xmax,ymax -> result[0], result[1], result[2], result[3]

    -> crop_img is the region of input image and it's located by bounding box and crop it out of image
    crop_img shows location of each text by 4 location (xmin, ymin, xmax, ymax)

    -> ocr is a text and it's extracted from images using Text Recognition model

    -> define ocr_format json with 6 variable (xmin, ymin, xmax, ymax, confidence) for preparing result

    """
    input_image = Image.open(io.BytesIO(file)).convert("RGB")
    results = model(input_image, 640)
    bbox = results.xyxy[0]
    ocr_ = []
    # Loc output of Machine Learning model ([bounding box (xmin,ymin,xmax,ymax) -> it's coordinate of text], confidence)
    for result in bbox:
        con = result[4]
        x1 = int(result[0])
        y1 = int(result[1])
        x2 = int(result[2])
        y2 = int(result[3])

        crop_img = input_image.crop((x1, y1, x2, y2))

        ocr = detector.predict(crop_img)

        ocr_format = {'recognizer': str(ocr),
                      'xmin': str(x1),
                      'ymin': str(y1),
                      'xmax': str(x2),
                      'ymax': str(y2),
                      'confidence': float(con)
                      }
        ocr_.append(ocr_format)

    return ocr_
