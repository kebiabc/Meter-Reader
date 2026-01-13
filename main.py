import io
import cv2
import uvicorn
import numpy as np
from typing import List
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 加载 YOLO 模型
model = YOLO("runs/detect/finetune/weights/best.pt")

# 类别名称映射
class_names = ['.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/detect/")
async def detect_objects(files: List[UploadFile] = File(...)):
    all_detection_results = []

    for file in files:
        try:
            # 读取上传的图像内容
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 使用 YOLO 进行推理
            results = model(image)
            
            # 收集所有检测结果
            digits = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2  # 计算中心x坐标用于排序
                digit = class_names[cls_id]
                digits.append({
                    "digit": digit,
                    "x_center": float(x_center),
                    "confidence": round(conf, 2)
                })
            
            # 按x坐标从左到右排序
            digits.sort(key=lambda x: x["x_center"])
            
            # 组合成最终读数
            reading = "".join([d["digit"] for d in digits])
            
            # 准备详细检测结果
            detection_details = [{
                "digit": d["digit"],
                "confidence": d["confidence"],
                "position": int(d["x_center"])
            } for d in digits]

            all_detection_results.append({
                "filename": file.filename,
                "reading": reading,
                "digits": detection_details
            })

        except Exception as e:
            all_detection_results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return all_detection_results

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, reload=False)
    # uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)  # 局域网访问：http://192.168.204.99:8080/