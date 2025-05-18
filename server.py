import copy
import torch

class FedServer:
    def __init__(self):
        from plate_recognition.plate_rec import init_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_recognizer = init_model(self.device, "weights/plate_rec_color.pth", is_color=True)

    def get_global_models(self):
        return {
            "recognizer": copy.deepcopy(self.global_recognizer.state_dict())
        }

    def aggregate(self, updates):
        # 按样本数加权聚合
        lens = [u.get("num_samples", 1) for u in updates]
        total = sum(lens)
        new_state = copy.deepcopy(updates[0]["recognizer"])
        for k in new_state.keys():
            if torch.is_floating_point(new_state[k]):
                new_state[k] = sum(u["recognizer"][k] * l for u, l in zip(updates, lens)) / total
            else:
                new_state[k] = updates[0]["recognizer"][k]
        self.global_recognizer.load_state_dict(new_state)

    def evaluate(self, test_set):
        from plate_recognition.plate_rec import get_plate_result
        self.global_recognizer.eval()
        device = self.device
        correct = 0
        total = 0
        for item in test_set:
            img = item["image"]
            x1, y1, x2, y2 = item["bbox"]
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            if len(roi.shape) == 2:
                import cv2
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            elif roi.shape[2] == 4:
                roi = roi[:, :, :3]
            plate, _, _, _ = get_plate_result(roi, device, self.global_recognizer, is_color=True)
            if plate == item["plate_number"]:
                correct += 1
            total += 1
        acc = correct / total if total > 0 else 0
        return acc