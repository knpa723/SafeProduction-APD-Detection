from vision_models.human.predictor import HumanPredictor

class HumanHandler:
    def __init__(self):
        self.HUMAN_MODEL_PATH = "src/models/human.pt"
        self.PPE_MODEL_PATH = "src/models/ppe_construction.pt"
        
        self.human_predictor = HumanPredictor(
            self.HUMAN_MODEL_PATH, 
            self.PPE_MODEL_PATH
        )

    def process_human_detections(self, frame):
        """
        Process 1 video frame for detections
        """
        return self.human_predictor.detect_humans(
            image=frame,
            conf=0.5
        )
        
    def process_human_tracking(self, frame):
        """
        Process 1 video frame for detection + tracking
        """
        return self.human_predictor.track_humans(
            image=frame,
            conf=0.5
        )