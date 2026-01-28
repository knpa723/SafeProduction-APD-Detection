from typing import List
from .model import HumanDetectionModel
from .tracker import HumanTracker
import cv2

class HumanPredictor:
    def __init__(self, human_model_path, ppe_model_path):
        self.human_model = HumanDetectionModel(human_model_path)
        self.ppe_model = HumanDetectionModel(ppe_model_path) 
        self.tracker = HumanTracker()

    def detect_humans(self, image, conf: float = 0.5) -> List[dict]:
        """
        Returns list of detected humans with bounding boxes
        """
        results = self.human_model.predict(image, conf=conf)
        
        detections = []
        
        for r in results:
            for box in r.boxes:
                score = float(box.conf[0])
                
                if score < conf:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = r.names[int(box.cls[0])]
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": score,
                    "label": label
                })
        
        return detections
    
    # def track_humans(self, image, conf: float = 0.5):
    #     result = self.human_model.predict(image, conf=conf)[0]
    #     tracked = self.tracker.update(frame=image, result=result)
    #     return tracked

    def track_humans(self, image, conf: float = 0.5):
        result = self.human_model.predict(image, conf=conf)[0]
        tracked = self.tracker.update(result, image)
        
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[max(0, y1):y2, max(0, x1):x2]
            
            if crop.size > 0:
                ppe_results = self.ppe_model.predict(crop, conf=0.4)
                ppe_labels = []
                for p_res in ppe_results:
                    for p_box in p_res.boxes:
                        label_name = p_res.names[int(p_box.cls[0])]
                        ppe_labels.append(label_name)
                
                if ppe_labels:
                    display_text = ", ".join(set(ppe_labels))
                    cv2.putText(
                        tracked, 
                        display_text, 
                        (x1, y1 - 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0, 255, 255), 
                        2
                    )
        
        return tracked