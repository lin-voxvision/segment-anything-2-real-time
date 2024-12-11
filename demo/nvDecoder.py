# This copyright notice applies to this file only
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys
import argparse
import numpy as np
import cv2

sys.path.append('../')
from pathlib import Path
import pycuda.driver as cuda
import PyNvVideoCodec as nvc
from Utils import cast_address_to_1d_bytearray


def decode_rtsp(gpu_id, rtsp_url, dec_file_path, use_device_memory):
    """
            函数用于解码RTSP流并将帧转换为RGB格式后保存为jpg图像。

            该函数将从RTSP流读取数据并将其分割成数据包(packets)。
            每个数据包包含符合annex.b标准的一帧基本比特流。
            数据包被发送到解码器进行解析和硬件加速解码。解码器返回可迭代的原始YUV帧列表。

            参数:
            - gpu_id (int): 要使用的GPU序号 [参数未使用]
            - rtsp_url (str): RTSP流地址
            - dec_file_path (str): 输出jpg图像的路径
            - use_device_memory (int): 如果设为1,输出解码帧为包装在CUDA Array Interface中的CUDeviceptr,
                                     否则为主机内存

            返回值: None

            示例:
            >>> decode_rtsp(0, "rtsp://example.com/stream", "path/to/output.jpg", 1)
    """
    # 创建RTSP流的解复用器
    nv_dmx = nvc.CreateDemuxer(filename=rtsp_url)
    nv_dec = nvc.CreateDecoder(gpuid=0,
                              codec=nv_dmx.GetNvCodecId(),
                              cudacontext=0,
                              cudastream=0,
                              usedevicememory=use_device_memory)

    decoded_frame_size = 0
    raw_frame = None
    width = nv_dmx.Width()
    height = nv_dmx.Height()

    seq_triggered = False
    print("FPS = ", nv_dmx.FrameRate())
    
    try:
        timeout = 0
        while True:  # 持续读取RTSP流
            for packet in nv_dmx:
                if packet is None:
                    timeout += 1
                    if timeout > 100:  # 设置超时阈值
                        print("无法获取RTSP流数据，请检查RTSP地址是否正确")
                        return
                    continue
                    
                for decoded_frame in nv_dec.Decode(packet):
                    if not seq_triggered:
                        decoded_frame_size = nv_dec.GetFrameSize()
                        raw_frame = np.ndarray(shape=decoded_frame_size, dtype=np.uint8)
                        seq_triggered = True

                    luma_base_addr = decoded_frame.GetPtrToPlane(0)
                    if use_device_memory:
                        cuda.memcpy_dtoh(raw_frame, luma_base_addr)
                        # 将NV12格式的帧重塑为正确的形状
                        yuv = raw_frame.reshape((height * 3 // 2, width))
                    else:
                        new_array = cast_address_to_1d_bytearray(base_address=luma_base_addr, size=decoded_frame.framesize())
                        yuv = np.frombuffer(new_array, dtype=np.uint8).reshape((height * 3 // 2, width))
                    
                    # 转换为BGR格式
                    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
                    # 保存为jpg图像
                    cv2.imwrite(dec_file_path, bgr)
                    
                    print("成功解码一帧")
                    return  # 如果只需要一帧就可以返回
                    
    except Exception as e:
        print(f"解码过程出错: {str(e)}")
    finally:
        print("正在停止RTSP流解码...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "此示例程序演示了RTSP流的解复用和解码。"
    )
    parser.add_argument(
        "-g", "--gpu_id", type=int, help="GPU ID, 查看nvidia-smi, 仅用于演示", )
    parser.add_argument(
        "-i", "--rtsp_url", type=str, required=True,
        help="RTSP流地址", )
    parser.add_argument(
        "-o", "--raw_file_path", required=True, type=Path, help="输出jpg图像路径", )
    parser.add_argument(
        "-d", "--use_device_memory", required=True, type=int, help="解码器输出表面在设备内存中,否则在主机内存中", )
    args = parser.parse_args()
    decode_rtsp(args.gpu_id, args.rtsp_url,
              args.raw_file_path.as_posix(),
              args.use_device_memory)
