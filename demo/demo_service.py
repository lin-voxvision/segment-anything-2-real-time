import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional
import asyncio
from fastapi import BackgroundTasks
import socket
import time



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
click_point = None
if_init = False
should_process = True
active_predictor_id = 1  # 当前激活的predictor编号

# UDP socket设置
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
UDP_IP = "192.168.0.114"
UDP_PORT = 8001

# 使用 bfloat16
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor

sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

# 初始化两个predictor
predictor1 = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
predictor2 = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
print("初始化完成两个predictor")

cap = cv2.VideoCapture("rtsp://admin:admin@127.0.0.1:8554/cam/realmonitor?channel=1&subtype=0&unicast=true")
width = 1920
height = 1080

output_path = "output.mp4"
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def switch_predictor():
    global active_predictor_id, predictor1, predictor2
    # 切换predictor
    active_predictor_id = 2 if active_predictor_id == 1 else 1
    # 重新实例化非活动的predictor
    if active_predictor_id == 1:
        predictor2 = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
    else:
        predictor1 = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

@app.post("/click_point/{x}/{y}")
async def set_click_point(x: int, y: int):
    global click_point, if_init
    click_point = (x, y)
    if if_init:  # 如果已经在跟踪，则切换predictor
        switch_predictor()
        if_init = False
    return {"message": f"点击坐标设置为 ({x}, {y})"}

@app.post("/cancel")
async def cancel_point():
    global click_point, if_init
    click_point = None
    if if_init:  # 如果已经在跟踪，则切换predictor
        switch_predictor()
        if_init = False
    return {"message": "取消点击坐标"}

@app.on_event("startup")
async def startup_event():
    # 在启动时开始处理视频
    asyncio.create_task(process_video())

async def process_video():
    global if_init, click_point, should_process, active_predictor_id
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        try:
            while should_process:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 获取当前活动的predictor
                current_predictor = predictor1 if active_predictor_id == 1 else predictor2

                if not if_init and click_point is not None:
                    ptx, pty = click_point
                    current_predictor.load_first_frame(frame)
                    if_init = True

                    ann_frame_idx = 0
                    ann_obj_id = 1

                    points = np.array([[ptx, pty]], dtype=np.float32)
                    labels = np.array([1], dtype=np.int32)

                    _, out_obj_ids, out_mask_logits = current_predictor.add_new_prompt(
                        frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
                    )

                elif if_init and click_point is not None:
                    start_time = time.time()
                    out_obj_ids, out_mask_logits = current_predictor.track(frame)
                    end_time = time.time()
                    print(f"track time: {end_time - start_time}")

                    start_time = time.time()
                    # 收集所有bbox信息
                    all_mask = np.zeros((height, width, 1), dtype=np.uint8)
                    bbox_data = []
                    for i in range(0, len(out_obj_ids)):
                        out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
                        all_mask = cv2.bitwise_or(all_mask, out_mask)
                        contours, _ = cv2.findContours(out_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            bbox_data.extend([x, y, w, h])

                    # 构造UDP消息
                    if bbox_data:
                        msg = f"{len(out_obj_ids)}," + ",".join(map(str, bbox_data))
                        udp_socket.sendto(msg.encode(), (UDP_IP, UDP_PORT))

                    end_time = time.time()
                    print(f"send udp time: {end_time - start_time}")

                    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
                    frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.write(frame)
                
                # 添加一个小延迟，让出CPU时间给其他任务
                await asyncio.sleep(0.001)

        except Exception as e:
            print(f"Error in video processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            out.release()
            udp_socket.close()

@app.on_event("shutdown")
async def shutdown_event():
    global should_process
    should_process = False
    cap.release()
    out.release()
    udp_socket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.0.126", port=8000)
