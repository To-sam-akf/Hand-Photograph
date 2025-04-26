import cv2
import numpy as np
import copy
import math

# parameters
# cap_region_x_begin = 0.6  # start point/total width
# cap_region_y_end = 0.6  # start point/total width
# threshold = 70  #  BINARY threshold
# blurValue = 41  # GaussianBlur parameter（41*41） 用于减少图像噪声，使手势轮廓更平滑
# bgSubThreshold = 50
# learningRate = 0.5  # 背景学习速率 范围0-1，0表示完全不更新背景
# bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

# variables
# isBgCaptured = 1  # bool, whether the background captured
# triggerSwitch = True  # if true, keyborad simulator works


def printThreshold(thr):
    print("! Changed threshold to " + str(thr))


# 分离前景（手部）和背景
# 去除噪声
# 只保留有用的前景部分（手部区域）
# 便于后续的手势检测和分析
def removeBG(frame,bgModel, learningRate=0):
    fgmask = bgModel.apply(frame, learningRate=learningRate)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    # bitwise_and将原始帧与掩码进行按位与运算，保留掩码中白色区域对应的原始图像部分
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def calculateFingers(res, drawing):  # -> 返回值：是否完成计算, 手指数量
    # 计算凸包缺陷来识别手指
    # res: 手部轮廓点集
    # drawing: 用于绘制结果的图像

    # 获取轮廓的凸包，returnPoints=False返回凸包顶点的索引而不是坐标点
    hull = cv2.convexHull(res, returnPoints=False)

    # 确保凸包点数足够进行后续处理
    if len(hull) > 3:
        # 计算凸包缺陷，得到凹陷区域的信息
        defects = cv2.convexityDefects(res, hull)

        # 确保检测到凸包缺陷
        if type(defects) != type(None):

            # 计数器，用于统计手指数量
            cnt = 0

            # 遍历所有凸包缺陷
            for i in range(defects.shape[0]):
                # 获取缺陷的起点(s)、终点(e)、最远点(f)和到最远点的距离(d)
                s, e, f, d = defects[i][0]

                # 转换点坐标格式
                start = tuple(res[s][0])  # 起点坐标
                end = tuple(res[e][0])  # 终点坐标
                far = tuple(res[f][0])  # 最远点坐标

                # 使用三边长度计算夹角
                # 计算三条边的长度
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

                # 使用余弦定理计算夹角
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

                # 如果夹角小于90度，认为是手指间的凹陷
                if angle <= math.pi / 2:
                    cnt += 1  # 手指数量加1
                    # 在最远点画一个圆，标记手指间隙
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)

            return True, cnt  # 返回计算成功和手指数量

    # 如果凸包点数不足或未检测到手指，返回失败
    return False, 0

def processForFingerDetection(frame, cap_region_x_begin=0.6, cap_region_y_end=0.6, threshold=70, blurValue=41, bgModel=None):
    img = removeBG(frame,bgModel, learningRate=0)
    img = img[0:int(cap_region_y_end * frame.shape[0]),
          int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
    # cv2.imshow('mask', img)

    # convert the image into binary image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    # cv2.imshow('blur', blur)
    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow('ori', thresh)

    # get the contours
    thresh1 = copy.deepcopy(thresh)
    return img, thresh1

# # Camera
# camera = cv2.VideoCapture(0)
# camera.set(10, 200)
# cv2.namedWindow('trackbar') # 创建一个新滑块窗口
# cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)
# bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
# while camera.isOpened():
#     ret, frame = camera.read()
#     threshold = cv2.getTrackbarPos('trh1', 'trackbar')
#     # 好的，这行代码 frame = cv2.bilateralFilter(frame, 5, 50, 100) 是使用 OpenCV 的双边滤波 (Bilateral Filter) 功能来处理图像 frame。
#     # 双边滤波是一种非线性的图像滤波方法，它在进行降噪的同时，能够非常好地保留图像的边缘。
#     frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
#     # frame = cv2.flip(frame, 1)  # 镜像
#     cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
#                   (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
#     cv2.imshow('original', frame)
#
#     #  Main operation
#     if isBgCaptured == 1:  # this part wont run until background captured
#         img, thresh = processForFingerDetection(frame)
#         thresh1 = copy.deepcopy(thresh)
#         contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 修改解包逻辑
#         length = len(contours)
#         maxArea = -1
#         if length > 0:
#             for i in range(length):  # find the biggest contour (according to area)
#                 temp = contours[i]
#                 area = cv2.contourArea(temp)
#                 if area > maxArea: # 找到最大轮廓的索引
#                     maxArea = area
#                     ci = i
#
#             # 获取最大轮廓
#             res = contours[ci]
#             hull = cv2.convexHull(res)
#             drawing = np.zeros(img.shape, np.uint8)
#             cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
#             cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
#
#             isFinishCal, cnt = calculateFingers(res, drawing)
#
#             # 触发开关
#             if triggerSwitch is True:
#                 if isFinishCal is True and 0< cnt <= 4:
#                     print(cnt)
#
#
#
#         cv2.imshow('output', drawing)
#
#     # Keyboard OP
#     k = cv2.waitKey(10)
#     if k == 27:  # press ESC to exit
#         camera.release()
#         cv2.destroyAllWindows()
#         break
#     elif k == ord('r'):  # press 'r' to reset the background
#         bgModel = None
#         triggerSwitch = False
#         isBgCaptured = 0
#         print('!!!Reset BackGround!!!')
#     if isBgCaptured == 0:
#         isBgCaptured =1
#         bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
