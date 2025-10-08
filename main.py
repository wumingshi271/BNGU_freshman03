import math

import cv2
import numpy as np

if __name__ == '__main__':
    # 创建 VideoCapture 对象，读取视频文件
    cap = cv2.VideoCapture('VID_20251003154949393.mp4')

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # 读取视频帧
    while True:
        ret, frame = cap.read()

        #获取视频每一帧的hsv值
        cap_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cap_h,cap_s,cap_v = cv2.split(cap_hsv)
        # 阈值化处理，提取黑色
        mask_h = cv2.inRange(cap_h,0, 180)
        mask_s = cv2.inRange(cap_s, 0, 255)
        mask_v = cv2.inRange(cap_v, 0, 60)
        # 对hsv的mask取与操作
        mask =cv2.bitwise_and(mask_h, mask_v, mask_s)
        # 在mask的位置对两个图像取与，获得特定的图像
        cap_out = cv2.bitwise_and(frame, frame, mask=mask)

        # 获取灰度图像
        gray = cv2.cvtColor(cap_out, cv2.COLOR_BGR2GRAY)
        #去除小噪点，使轮廓更加完整
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(gray, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        # 查找所有轮廓
        countours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in countours:
            # 过滤掉面积小的轮廓
            area = cv2.contourArea(cnt)
            if area < 20000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


        # 如果读取到最后一帧，退出循环
        if not ret:
            break

        # 显示当前帧
        cv2.imshow('Video', frame)

        # 按下 'q' 键退出
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
