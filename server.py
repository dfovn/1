import copy
import torch
from tqdm import tqdm
from detect_rec_plate import load_model, det_rec_plate
from plate_recognition.plate_rec import init_model


class FedServer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Server using device: {self.device}")

        # 完全使用原有模型加载方式
        self.global_detector = load_model("weights/yolov8s.pt", self.device)
        self.global_recognizer = init_model(self.device, "weights/plate_rec_color.pth", is_color=True)

        # 冻结YOLO基础层（保持原有检测模型特性）
        for name, param in self.global_detector.named_parameters():
            if 'model.24' not in name:  # 仅训练检测头
                param.requires_grad = False
            else:
                param.requires_grad = True

    def aggregate(self, client_updates):
        """改进的聚合逻辑，保持原有模型结构"""
        # 检测模型聚合
        detector_state = copy.deepcopy(self.global_detector.state_dict())
        for key in detector_state:
            if 'model.24' in key:
                detector_state[key] = torch.mean(
                    torch.stack([update['detector'][key] for update in client_updates]), dim=0)
        self.global_detector.load_state_dict(detector_state)

        # 识别模型聚合（全参数平均）
        recognizer_state = copy.deepcopy(self.global_recognizer.state_dict())
        for key in recognizer_state:
            if key in client_updates[0]['recognizer']:
                recognizer_state[key] = torch.mean(
                    torch.stack([update['recognizer'][key] for update in client_updates]), dim=0)
        self.global_recognizer.load_state_dict(recognizer_state)

    def get_global_models(self):
        return {
            "detector": copy.deepcopy(self.global_detector.state_dict()),
            "recognizer": copy.deepcopy(self.global_recognizer.state_dict())
        }

    def evaluate(self, dataset):
        """使用原有检测识别流程进行评估"""
        self.global_detector.eval()
        self.global_recognizer.eval()

        correct = 0
        total = 0
        for item in tqdm(dataset, desc="Evaluating"):
            try:
                # 使用原有检测识别流程
                result_list = det_rec_plate(
                    item['image'],
                    item['image'].copy(),
                    self.global_detector,
                    self.global_recognizer
                )
                if result_list:
                    pred = result_list[0]
                    if (pred['plate_no'] == item['plate_number'] and
                            pred['plate_color'] == item['plate_color']):
                        correct += 1
                total += 1
            except Exception as e:
                print(f"Evaluation error: {e}")
                continue
        return correct / total if total > 0 else 0.0