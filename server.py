import copy

import cv2
import torch
from plate_recognition.plate_rec import init_model, image_processing, get_plate_result


class FedServer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_recognizer = init_model(self.device, "weights/plate_rec_color.pth", is_color=True)

    def get_global_models(self):
        return {"recognizer": copy.deepcopy(self.global_recognizer.state_dict())}

    def aggregate(self, updates):
        # FedAvg 聚合识别模型
        new_state = copy.deepcopy(updates[0]["recognizer"])
        for k in new_state.keys():
            if torch.is_floating_point(new_state[k]):
                for update in updates[1:]:
                    new_state[k] += update["recognizer"][k].to(new_state[k].dtype)
                new_state[k] /= len(updates)
            else:
                # 非浮点型参数（如int、long），直接取第一个
                new_state[k] = updates[0]["recognizer"][k]
        self.global_recognizer.load_state_dict(new_state)

    def evaluate(self, testset):
        self.global_recognizer.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for item in testset:
                img = item["image"]
                x1, y1, x2, y2 = item["bbox"]
                roi = img[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                if len(roi.shape) == 2:
                    roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                elif roi.shape[2] == 4:
                    roi = roi[:, :, :3]
                processed = image_processing(roi, self.device)
                plate, _, color, _ = get_plate_result(roi, self.device, self.global_recognizer, is_color=True)
                if plate == item["plate_number"]:
                    correct += 1
                total += 1
        return correct / total if total else 0.0