import cv2
import torch
import copy
from plate_recognition.plate_rec import init_model
from utils import PlateDataset

class FedServer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_recognizer = init_model(self.device, "weights/plate_rec_color.pth", is_color=True)

    def get_global_models(self):
        return {
            "recognizer": copy.deepcopy(self.global_recognizer.state_dict())
        }

    def aggregate(self, client_updates):
        # 按样本数加权平均
        total = sum(u["num_samples"] for u in client_updates)
        global_state = copy.deepcopy(self.global_recognizer.state_dict())
        for k in global_state.keys():
            global_state[k] = sum(
                u["recognizer"][k].cpu() * (u["num_samples"] / total)
                for u in client_updates
            )
        self.global_recognizer.load_state_dict(global_state)

    def evaluate(self, test_set):
        self.global_recognizer.eval()
        correct = 0
        total = 0

        char_total = 0
        char_correct = 0

        color_total = 0
        color_correct = 0

        with torch.no_grad():
            for item in test_set:
                img = item["image"]
                x1, y1, x2, y2 = item["bbox"]
                plate_number = item["plate_number"]
                plate_color = item["plate_color"]
                roi = img[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                if len(roi.shape) == 2:
                    roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                elif roi.shape[2] == 4:
                    roi = roi[:, :, :3]
                from plate_recognition.plate_rec import image_processing, decodePlate, color
                input_tensor = image_processing(roi, self.device)
                logits, color_logits = self.global_recognizer(input_tensor)
                if logits.dim() == 3:
                    logits = logits.permute(1, 0, 2)
                preds = logits.softmax(2).argmax(2).squeeze(1).detach().cpu().numpy()
                pred_chars, _ = decodePlate(preds)
                plateName = "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
                pred_str = ''.join([plateName[i] for i in pred_chars if i < len(plateName)])
                # 完全匹配准确率
                if pred_str == plate_number:
                    correct += 1
                total += 1

                # 字符级准确率
                min_len = min(len(pred_str), len(plate_number))
                char_total += len(plate_number)
                char_correct += sum([pred_str[i] == plate_number[i] for i in range(min_len)])

                # 颜色准确率
                pred_color_idx = color_logits.argmax(dim=-1).item()
                pred_color = color[pred_color_idx]
                if pred_color == plate_color:
                    color_correct += 1
                color_total += 1

        full_acc = correct / total if total > 0 else 0.0
        char_acc = char_correct / char_total if char_total > 0 else 0.0
        color_acc = color_correct / color_total if color_total > 0 else 0.0

        print(f"[评估] 完全匹配率: {full_acc:.2%}, 字符级准确率: {char_acc:.2%}, 颜色准确率: {color_acc:.2%}")
        return {
            "full_accuracy": full_acc,
            "char_accuracy": char_acc,
            "color_accuracy": color_acc
        }