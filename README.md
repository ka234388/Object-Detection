# Object-Detection
Faster R-CNN, DETR, Grounding-DINO
Code is running object detection across the full COCO val2017 dataset (5000 images). And for each image performed I run inference through Faster R-CNN, DETR, Grounding-Dino and Dino model and collect detections based on confidence threshold then save bounding boxes, scores, labels for each detection. I’m measuring inference time. Then after all images summarizing the results which are total detections, detections per image, mean confidence, avg latency, FPS saved in JSON results file.
Object Detection Benchmark – COCO Val2017

Course: CAP 6411 – Computer Vision Systems
Author: Karthika Ramasamy

🧠 Overview

This project runs object detection across the full COCO val2017 dataset (5,000 images) using multiple state-of-the-art models:

Faster R-CNN (ResNet-50 FPN)

DETR (Transformer-based detection)

Grounding DINO (Open-vocabulary model)

(DINO model in progress due to build issues)

Each image is passed through these detectors; detections are filtered by confidence threshold, and the following metrics are saved:

Bounding boxes, scores, and labels

Inference time per image

Total detections and detections per image

Mean confidence and average latency (FPS)

Results are summarized and stored in a JSON file.

⚙️ Setup
git clone https://github.com/<your_username>/<your_repo>.git
cd <your_repo>
pip install -r requirements.txt
python run_inference.py

📊 Results Summary
Model	Avg Time / Image	FPS	Avg Detections / Img	Mean Confidence
Grounding DINO	422 ms	2.37	4.3	0.51
Faster R-CNN	62 ms	16.1	9.0	0.816
DETR	45 ms	22.2	11.5	0.84
🔍 Model Insights

Faster R-CNN

Strong anchor-based detections

~9 objects/image with high confidence (0.82)

Reliable and steady predictions

DETR

Transformer architecture without anchors

Fastest (22 FPS), most confident (0.84)

Broad coverage including small objects

Grounding DINO

Open-vocabulary prompt-based detection

Slower (~2 FPS) with lower confidence (0.51)

Best suited for text-guided tasks beyond COCO classes

🧩 Comparative Highlights

Speed: DETR > Faster R-CNN > Grounding DINO

Confidence: DETR ≈ Faster R-CNN > Grounding DINO

Class Variety: DETR detects widest range of COCO categories

Open-Vocabulary Capability: Grounding DINO unique advantage

🖼️ Qualitative Examples

Faster R-CNN: Accurate boxes around laptops, monitors, keyboards, persons.

DETR: Sharper boxes and near-1.0 confidence in indoor scenes.

Grounding DINO: Detects persons and vehicles at crosswalks with moderate confidence.

⚠️ Issues Encountered

DINO model faced build issues during inference loading.

Work in progress to resolve model dependency errors.

📁 Outputs

results.json → summarized metrics

detections/ → visualized bounding box images

plots/ → confidence and latency histograms

🧾 Conclusion

DETR achieved the best balance of speed, confidence, and recall.
Faster R-CNN offered reliable accuracy at moderate speed.
Grounding DINO was slow but valuable for text-guided detection.
