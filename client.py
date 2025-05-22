import os
import time
import torch
import copy
import cv2
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from plate_recognition.plate_rec import image_processing
from utils import PlateDataset
from plate_recognition.plate_rec import init_model
from ultralytics.nn.tasks import attempt_load_weights

def load_model(weights, device):  # 加载yolov8 模型
    model = attempt_load_weights(weights, device=device)  # load FP32 model
    return model
def collate_fn(batch):
    return batch

def add_dp_noise(state_dict, noise_scale=1e-3):
    noisy_state = copy.deepcopy(state_dict)
    for k, v in noisy_state.items():
        if torch.is_floating_point(v):
            noise = torch.normal(0, noise_scale, size=v.shape, device=v.device)
            noisy_state[k] = v + noise
    return noisy_state

def xywh2xyxy(box, img_w, img_h):
    # [x_center, y_center, w, h] (normalized) -> [x1, y1, x2, y2] (pixel)
    x, y, w, h = box
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]

def iou_xyxy(box1, box2):
    # box: [x1, y1, x2, y2]
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

class FedClient:
    def __init__(self, client_id, data_root, server, dp_noise_scale=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_id = client_id
        self.server = server
        self.dp_noise_scale = dp_noise_scale

        self.local_detector = load_model("weights/yolov8s.pt", self.device)
        self.local_detector.eval()  # detector不训练
        self.local_recognizer = init_model(self.device, "weights/plate_rec_color.pth", is_color=True)
        self.dataset = PlateDataset(data_root, client_id=client_id, mode='train')
        self.loader = DataLoader(self.dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)

        self.plateName = "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
        self.color_map = {'黑色': 0, '蓝色': 1, '绿色': 2, '白色': 3, '黄色': 4}

    def detect_plate_boxes(self, img):
        # 检测模型推理
        img_h, img_w = img.shape[:2]
        from detect_rec_plate import pre_processing, post_processing
        img_in, r, left, top = pre_processing(img, self.device)
        with torch.no_grad():
            prediction = self.local_detector(img_in)[0]
            outputs = post_processing(prediction, conf=0.3, iou_thresh=0.5, r=r, left=left, top=top)
        det_boxes = []
        for out in outputs:
            box = [int(x) for x in out[:4]]
            det_boxes.append(box)
        return det_boxes

    def local_train(self, epochs=5):
        global_params = self.server.get_global_models()
        self.local_recognizer.load_state_dict(global_params["recognizer"])

        optimizer = torch.optim.Adam(self.local_recognizer.parameters(), lr=1e-4)
        criterion_plate = nn.CTCLoss(blank=0, zero_infinity=True)
        criterion_color = nn.CrossEntropyLoss()
        iou_thresh = 0.5  # 检测框和GT匹配阈值

        for epoch in range(epochs):
            self.local_recognizer.train()
            epoch_loss = 0.0
            valid_batches = 0
            for batch in self.loader:
                imgs, targets, colors, target_lens = [], [], [], []
                for item in batch:
                    img = item["image"]
                    gt_box = item["bbox"]
                    gt_number = item["plate_number"]
                    gt_color = item["plate_color"]
                    img_h, img_w = img.shape[:2]

                    # 1. 用检测模型检测所有车牌框
                    det_boxes = self.detect_plate_boxes(img)
                    # 2. 跟label中的GT框做IoU匹配，选最大IoU的框
                    max_iou = 0
                    best_box = None
                    for det in det_boxes:
                        iou = iou_xyxy(gt_box, det)
                        if iou > max_iou:
                            max_iou = iou
                            best_box = det
                    if max_iou < iou_thresh or best_box is None:
                        continue  # 检测没找到有效GT，跳过

                    x1, y1, x2, y2 = best_box
                    roi = img[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    if len(roi.shape) == 2:
                        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                    elif roi.shape[2] == 4:
                        roi = roi[:, :, :3]
                    processed = image_processing(roi, self.device)
                    idxs = self._str_to_indices(gt_number)
                    imgs.append(processed)
                    targets.append(idxs)
                    colors.append(self._color_to_idx(gt_color))
                    target_lens.append(len(idxs))
                if not imgs:
                    continue
                imgs = torch.cat(imgs, dim=0)
                logits, color_logits = self.local_recognizer(imgs)
                if logits.dim() == 3:
                    logits = logits.permute(1, 0, 2)
                else:
                    raise RuntimeError("Unexpected logits shape")
                targets_1d = torch.cat(targets)
                input_lengths = torch.full((imgs.shape[0],), logits.size(0), dtype=torch.long, device=self.device)
                target_lengths = torch.tensor(target_lens, dtype=torch.long, device=self.device)
                color_targets = torch.tensor(colors, dtype=torch.long, device=self.device)
                try:
                    loss_plate = criterion_plate(logits.log_softmax(2), targets_1d, input_lengths, target_lengths)
                    loss_color = criterion_color(color_logits, color_targets)
                    loss = loss_plate + loss_color
                except Exception as e:
                    print(f"[ERROR] Loss computation error: {e}")
                    continue
                if not torch.isfinite(loss) or loss.item() < 0:
                    print(f"[WARN] Negative or NaN loss detected! Skipping batch.")
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                valid_batches += 1
            print(f"Client {self.client_id} Epoch {epoch+1} Loss: {epoch_loss/(valid_batches or 1):.4f}")
        # 差分隐私：在本地模型参数上添加高斯噪声
        noisy_state = add_dp_noise(self.local_recognizer.state_dict(), noise_scale=self.dp_noise_scale)
        return {
            "recognizer": noisy_state,
            "num_samples": len(self.dataset)
        }

    def _str_to_indices(self, s):
        idxs = []
        for ch in s:
            if ch in self.plateName:
                idx = self.plateName.index(ch)
                idxs.append(idx)
            else:
                idxs.append(0)
        return torch.tensor(idxs, dtype=torch.long, device=self.device)

    def _color_to_idx(self, color):
        return self.color_map.get(color, 0)