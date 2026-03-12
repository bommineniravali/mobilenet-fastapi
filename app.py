from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import onnxruntime as ort
import numpy as np
import cv2
import uuid

app = FastAPI()

# Load model
MODEL_PATH = "ssd_mobilenet_v1.onnx"
session = ort.InferenceSession(MODEL_PATH)

input_name = session.get_inputs()[0].name

@app.get("/")
def home():
    return {"message": "Object Detection API  Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    h, w, _ = img.shape

    # preprocessing
    input_img = cv2.resize(img, (300, 300))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = np.expand_dims(input_img, axis=0).astype(np.uint8)

    outputs = session.run(None, {input_name: input_img})

    boxes = outputs[1][0]
    classes = outputs[2][0]
    scores = outputs[3][0]

    for i in range(len(scores)):
        if scores[i] > 0.5:

            ymin, xmin, ymax, xmax = boxes[i]

            x1 = int(xmin * w)
            y1 = int(ymin * h)
            x2 = int(xmax * w)
            y2 = int(ymax * h)

            cv2.rectangle(img, (x1,y1), (x2,y2),
                          (0,255,0), 2)

            label = f"ID:{int(classes[i])} {scores[i]:.2f}"

            cv2.putText(
                img,
                label,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

    output_name = f"output_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(output_name, img)

    return FileResponse(output_name)
