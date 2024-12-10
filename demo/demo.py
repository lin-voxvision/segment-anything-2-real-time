import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor
import time


sam2_checkpoint = "../checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
print(predictor)

cap = cv2.VideoCapture("rtsp://admin:admin@127.0.0.1:8554/cam/realmonitor?channel=1&subtype=0&unicast=true")


width = 1920
height = 1080


# 设置输出视频参数
output_path = "output.mp4"
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


if_init = False

ptx = 960
pty = 540


i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if not if_init:

        predictor.load_first_frame(frame)
        if_init = True

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        # Let's add a positive click at (x, y) = (210, 350) to get started


        ##! add points, `1` means positive click and `0` means negative click
        points = np.array([[ptx, pty]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        print("points, labels", points, labels)

        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
        )

        # ## ! add bbox
        # bbox = np.array([[600, 214], [765, 286]], dtype=np.float32)
        # _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
        # )

        ##! add mask
        # mask_img_path="../notebooks/masks/aquarium/aquarium_mask.png"
        # mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        # mask = mask / 255

        # _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, mask=mask
        # )

    else:
        out_obj_ids, out_mask_logits = predictor.track(frame)

        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        # print(all_mask.shape)
        for i in range(0, len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255

            all_mask = cv2.bitwise_or(all_mask, out_mask)

        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
        frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
        cv2.rectangle(frame, (ptx - 10, pty - 10), (ptx + 10, pty + 10), (0, 0, 255), 2)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out.write(frame)
    # cv2.imshow("frame", frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
        # break

cap.release()
# gif = imageio.mimsave("./result.gif", frame_list, "GIF", duration=0.00085)
