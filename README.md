# Object-Detection
Faster R-CNN, DETR, Grounding-DINO
Code is running object detection across the full COCO val2017 dataset (5000 images). And for each image performed I run inference through Faster R-CNN, DETR, Grounding-Dino and Dino model and collect detections based on confidence threshold then save bounding boxes, scores, labels for each detection. I’m measuring inference time. Then after all images summarizing the results which are total detections, detections per image, mean confidence, avg latency, FPS saved in JSON results file.
