import cv2
import numpy as np
from colorDetection import colorCatagory, centerFrame, centerHSVAnalysis, generateCollection, create_color_blocks, centerProcess
from yolo import YOLO
import time
import os


def colorDetection(frame, index):
    center_frame, offset_x, offset_y = centerFrame(frame)

    center_frame_GW = centerProcess(center_frame)

    center_hsv = cv2.cvtColor(center_frame_GW, cv2.COLOR_BGR2HSV)



    # 获得主导颜色的HSV
    dominant_color = centerHSVAnalysis(center_hsv)

    # 生成配色方案
    color_schemes = generateCollection(dominant_color)
    # 根据配色方案生成色块
    create_color_blocks(color_schemes, index)

    # 进行颜色检测，同时更新帧
    detected_color, frame = colorCatagory(center_hsv, frame, offset_x, offset_y)

    # 显示结果帧
    cv2.imshow("Color Detection", frame)


if __name__ == '__main__':
    device = 0
    size = 416
    confidence = 0.2
    hands = -1

    # 时间间隔控制参数
    last_detection_time = 0
    TIME_INTERVAL = 0.0  # 设置间隔时间（秒
    timegethand = time.time()
    getcolor = 1
    DetectionColorDelay = 2
    HandDetectionDelay = 2

    yolo = YOLO("./models/cross-hands-tiny.cfg", "./models/cross-hands-tiny.weights", ["hand"])
    yolo.size = int(size)
    yolo.confidence = float(confidence)

    print("starting webcam...")
    cv2.namedWindow("preview")

    # 初始化摄像头
    vc = cv2.VideoCapture(device)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    # 图片索引
    index = 0
    while rval:
        width, height, inference_time, results = yolo.inference(frame)

        # display fps
        cv2.putText(frame, f'{round(1 / inference_time, 2)} FPS', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 2)

        # sort by confidence
        results.sort(key=lambda x: x[2])

        # how many hands should be shown
        hand_count = len(results)

        # Color Detection开关
        current_time = time.time()
        if current_time - last_detection_time > TIME_INTERVAL:

            # 更新检测时间
            last_detection_time = current_time
            # 开始检测
            width, height, inference_time, results = yolo.inference(frame)

            # display fps
            cv2.putText(frame, f'{round(1 / inference_time, 2)} FPS', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2)
            # sort by confidence
            results.sort(key=lambda x: x[2])

            # how many hands should be shown
            hand_count = len(results)

            # Color Detection开关

            if hand_count and current_time - timegethand > HandDetectionDelay:
                timegethand = current_time
                getcolor = 0

                if hands != -1:
                    hand_count = int(hands)  # 限制手的数量,控制画框数量

                # display hands
                for detection in results[:hand_count]:
                    id, name, confidence, x, y, w, h = detection
                    cx = x + (w / 2)
                    cy = y + (h / 2)

                    # draw a bounding box rectangle and label on the image
                    color = (0, 255, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "%s (%s)" % (name, round(confidence, 2))
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)

                print(f"Waiting for {DetectionColorDelay}.")

            if current_time - timegethand > DetectionColorDelay and getcolor == 0:
                getcolor = 1
                # 进行颜色检测
                colorDetection(frame, index)

                # 保存结果
                folder_dir = "./resPic"
                os.makedirs(folder_dir, exist_ok=True)  # 确保目录存在
                path = f"{folder_dir}/frame_{index}.jpg"
                cv2.imwrite(path, frame)
                index += 1

        cv2.imshow("preview", frame)
        rval, frame = vc.read()

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow("preview")
    vc.release()
