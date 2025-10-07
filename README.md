# Object-Detection
Faster R-CNN, DETR, Grounding-DINO
Code is running object detection across the full COCO val2017 dataset (5000 images). And for each image performed I run inference through Faster R-CNN, DETR, Grounding-Dino and Dino model and collect detections based on confidence threshold then save bounding boxes, scores, labels for each detection. I’m measuring inference time. Then after all images summarizing the results which are total detections, detections per image, mean confidence, avg latency, FPS saved in JSON results file.
Object Detection Benchmark – COCO Val2017

# 🧠 Object Detection Benchmark – COCO Val2017  

**Course:** CAP 6411 – Computer Vision Systems  
**Author:** Karthika Ramasamy  

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]() [![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)]() [![Dataset](https://img.shields.io/badge/Dataset-COCO%20val2017-orange.svg)]() [![Status](https://img.shields.io/badge/Models-DETR%20%7C%20Faster%20R--CNN%20%7C%20Grounding--DINO-green.svg)]()

---

## 🚀 Overview  
This project benchmarks multiple **object detection architectures** on the **COCO val2017** dataset (5,000 images).  
Each image is processed through different models and evaluated on **speed, confidence, and recall**.  

**Implemented Models:**  
- 🟦 Faster R-CNN (ResNet-50 FPN) – anchor-based  
- 🟪 DETR (Transformer-based, anchor-free)  
- 🟧 Grounding DINO (Open-vocabulary, text-prompt)  
- ⚪ DINO  

Each run records bounding boxes, confidence scores, and inference time.  
Aggregated metrics are stored in a `results.json` file.

---

## ⚙️ Setup  

```bash
git clone https://github.com/<your_username>/<your_repo>.git
cd <your_repo>
pip install -r requirements.txt
python run_inference.py
