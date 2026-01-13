# YOLOv8-Based Digit Recognition System (FastAPI Deployment)

This project implements a complete pipeline from training a digit detection model to deploying it via a web API. Users can upload images through a web page, and the system will automatically recognize the sequence of digits in the image.

---

## 🔧 Project Overview

- Uses **YOLOv8** for digit detection;
- Trains an initial model on a **public dataset**, then **fine-tunes** on real-world data;
- Built with **FastAPI** to support image upload and inference via a web service;
- Returns full digit readings along with the confidence and position of each digit.

---

## 🧠 Model Training Process

### 1. Initial Training (Using Public Dataset)

- Select a digit detection dataset (e.g., MNIST Detection, Synthetic Digits);
- Format the dataset in YOLOv8 format:
  ```
  ├── images/
  ├── labels/
  └── data.yaml
  ```
- Train the initial model with:
  ```bash
  yolo detect train \
      data=data.yaml \
      model=yolov8n.pt \
      epochs=100 \
      imgsz=640
  ```

---

### 2. Model Testing and Fine-Tuning

- Test the trained model on real-world samples:
  ```bash
  yolo detect predict \
      model=runs/detect/train/weights/best.pt \
      source=real_samples/
  ```
- Collect and label real-world data, then fine-tune the model:
  ```bash
  yolo detect train \
      data=real_data.yaml \
      model=runs/detect/train/weights/best.pt \
      epochs=50 \
      imgsz=640
  ```

---

## 🚀 Model Deployment (FastAPI Implementation)

### 1. Run the Server

The main program is in `main.py`. Run the service using:

```bash
python main.py
# or:
uvicorn main:app --host 127.0.0.1 --port 8080
```

### 2. Web Upload Interface

- Visit the root path `/` to access the upload page;
- Upload images and get digit recognition results.

---

### 3. Example Output Format

```json
[
  {
    "filename": "example.jpg",
    "reading": "123",
    "digits": [
      { "digit": "1", "confidence": 0.95, "position": 102 },
      { "digit": "2", "confidence": 0.93, "position": 150 },
      { "digit": "3", "confidence": 0.90, "position": 198 }
    ]
  }
]
```

---

## 📁 Project Structure

```
your_project/
├── main.py                      # FastAPI server
├── templates/
│   └── upload.html              # HTML upload form
├── static/                      # Static resources (CSS/JS)
├── runs/
│   └── detect/
│       └── finetune/
│           └── weights/
│               └── best.pt     # Fine-tuned YOLOv8 model
├── datasets/
│   ├── public_digits/           # Public dataset
│   └── real_samples/            # Real-world samples
└── data.yaml                    # YOLO dataset config file
```

---

## 📌 Dependencies

- Python 3.8+
- OpenCV
- numpy
- fastapi
- uvicorn
- ultralytics

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually install core packages:

```bash
pip install fastapi uvicorn opencv-python numpy ultralytics
```

---

## 📷 Example Display 

<img width="896" height="1920" alt="2" src="https://github.com/user-attachments/assets/a236dabe-2b1d-45d5-ac5a-e211587bd84c" />


---

## ✅ Future Improvements

- Display bounding boxes and results on the frontend;
- Add automatic model reloading and performance evaluation;
- Implement model versioning and logging;
- Enhance UI with drag-and-drop upload, batch preview, etc.
