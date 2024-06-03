# import numpy as np
# from fastapi import FastAPI,File,UploadFile
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import  Image
# import tensorflow as tf
# app=FastAPI()
#
# MODEL = tf.keras.models.load_model("../models1/2")
# CLASS_NAMES =["NORMAL","PNEUMONIA"]
#
# @app.get("/ping")
# async  def ping():
#     return "Hello"
#
# def read_file_as_image(data) -> np.ndarray:
#     image=np.array(Image.open(BytesIO(data)))
#     return image
#
# @app.post("/predict")
# async def predict(
#         file: UploadFile = File(...)
#
# ):
#     image = read_file_as_image(await file.read())
#     img_batch =np.expand_dims(image,0)
#     predictions = MODEL.predict(img_batch)
#     predicted_class= CLASS_NAMES[np.argmax(predictions[0])]
#     confidence=np.max(predictions[0])
#     return {
#         'class':predicted_class,
#         'confidence':float(confidence)
#     }
#
#
#
#
#
#
# if __name__ == "__main__":
#     uvicorn.run(app,host='localhost',port=8000)


import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# Load the Keras model from chest_xray.h5
MODEL = tf.keras.models.load_model("../models/your saved model")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGE_SIZE = (224, 224)  # Update the size to match the expected input shape of the model

@app.get("/ping")
async def ping():
    return "Hello"

def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
        image = image.resize(IMAGE_SIZE)  # Resize the image to the expected dimensions
        image = np.array(image) / 255.0
        return image
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading image: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {
            'class': predicted_class,
            'confidence': confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
