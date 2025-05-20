import cv2
import torch
import numpy as np
from utils import PlateDataset
from detect_rec_plate import load_model, pre_processing, post_processing
from plate_recognition.plate_rec import init_model, image_processing, decodePlate, color


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


def run_detect_rec_pipeline(det_weight, rec_weight, test_root, iou_thresh=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detect_model = load_model(det_weight, device)
    detect_model.eval()
    rec_model = init_model(device, rec_weight, is_color=True)
    rec_model.eval()
    plateName = "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
    dataset = PlateDataset(test_root, mode="test")

    total = 0
    det_found = 0
    full_match = 0
    char_total = 0
    char_match = 0
    color_total = 0
    color_match = 0

    for item in dataset:
        img = item["image"]
        gt_box = item["bbox"]
        gt_plate = item["plate_number"]
        gt_color = item["plate_color"]
        img_h, img_w = img.shape[:2]

        # 检测
        img_in, r, left, top = pre_processing(img, device)
        with torch.no_grad():
            prediction = detect_model(img_in)[0]
            outputs = post_processing(prediction, conf=0.3, iou_thresh=0.5, r=r, left=left, top=top)
        # 匹配检测框和GT
        best_iou = 0
        best_box = None
        for out in outputs:
            det_box = [int(x) for x in out[:4]]
            iou = iou_xyxy(gt_box, det_box)
            if iou > best_iou:
                best_iou = iou
                best_box = det_box
        total += 1
        if best_iou < iou_thresh or best_box is None:
            continue  # 未检测到
        det_found += 1
        x1, y1, x2, y2 = best_box
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        if len(roi.shape) == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        elif roi.shape[2] == 4:
            roi = roi[:, :, :3]
        # 识别
        input_tensor = image_processing(roi, device)
        with torch.no_grad():
            logits, color_logits = rec_model(input_tensor)
            if logits.dim() == 3:
                logits = logits.permute(1, 0, 2)
            preds = logits.softmax(2).argmax(2).squeeze(1).detach().cpu().numpy()
            pred_chars, _ = decodePlate(preds)
            pred_str = ''.join([plateName[i] for i in pred_chars if i < len(plateName)])
            pred_color_idx = color_logits.argmax(dim=-1).item()
            pred_color = color[pred_color_idx]
        # 评估
        if pred_str == gt_plate:
            full_match += 1
        cl = min(len(pred_str), len(gt_plate))
        char_total += len(gt_plate)
        char_match += sum([pred_str[i] == gt_plate[i] for i in range(cl)])
        color_total += 1
        if pred_color == gt_color:
            color_match += 1

    print(f"\n==== 检测-识别串行评估 ====")
    print(f"检测召回率: {det_found / total:.2%}  ({det_found}/{total})")
    print(f"完整识别准确率: {full_match / total:.2%}")
    print(f"字符级准确率: {char_match / char_total:.2%}")
    print(f"颜色准确率: {color_match / color_total:.2%}")


if __name__ == "__main__":
    det_weight = "weights/yolov8s.pt"
    rec_weight = "weights/plate_rec_color.pth"
    test_root = "./ccpd2019"  # 数据集根目录
    run_detect_rec_pipeline(det_weight, rec_weight, test_root)