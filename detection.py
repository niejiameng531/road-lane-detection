# Steve Mitchell
# June 2017

import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
#import argparse
import cv2
import imutils
import time

# preliminary attempt at lane following system
# largely derived from: https://medium.com/pharos-production/
# road-lane-recognition-with-opencv-and-ios-a892a3ab635c

# identify filename of video to be analyzed
cap = cv2.VideoCapture('testvideo2.mp4')

# loop through until entire video file is played
while(cap.isOpened()):

    # read video frame & show on screen
    # cap.read() 方法会返回一个布尔值和当前帧。布尔值表示是否成功读取到帧（ret），当前帧则存储在 frame 变量中
    ret, frame = cap.read()
    # cv2.imshow("Original Scene", frame)

    # snip section of video frame of interest & show on screen
    # frame 是通过 VideoCapture.read() 函数获取的当前帧，它是一个 NumPy 数组，表示帧的像素数据
    # 对 frame 进行切片操作，以获取一个子区域
    snip = frame[500:700,300:900]
    cv2.imshow("Snip",snip)

    # create polygon (trapezoid) mask to select region of interest
    # 创建一个梯形（或称为梯形状）的掩码（mask），用于选择图像中的感兴趣区域（Region of Interest, ROI)
    # snip.shape存储的是snip的高度和宽度
    mask = np.zeros((snip.shape[0], snip.shape[1]), dtype="uint8")
    pts = np.array([[25, 190], [275, 50], [380, 50], [575, 190]], dtype=np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    cv2.imshow("Mask", mask)

    # apply mask and show masked image on screen
    # 函数执行按位与操作，只有当掩码中的像素值为255（白色）时，对应的 snip 中的像素值才会被保留下来
    # 只有在mask中对应位置的像素值为时，才会对src1和src2中的像素进行按位与操作
    masked = cv2.bitwise_and(snip, snip, mask=mask)
    cv2.imshow("Region of Interest", masked)

    # convert to grayscale then black/white to binary image
    
    frame = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    # 定义了一个阈值 thresh，用于后续的二值化操作
    # 阈值是一个预定义的数值，用于将图像中的像素值分类为两个类别
    # 通常情况下，大于阈值的像素被归类为一类（例如白色或高亮度），小于或等于阈值的像素被归类为另一类（例如黑色或低亮度）
    thresh = 210
    # 将灰度图像 frame 转换为二值图像，大于这个阈值的像素将被设置为白色（值为255），小于或等于这个阈值的像素将被设置为黑色（值为0）
    # 函数返回一个包含两个元素的元组：第一个元素是阈值使用的实际值，第二个元素是二值化后的输出图像。
    # 在这个例子中，我们只关心输出图像，所以使用 [1] 来获取它，并将其赋值给 frame
    frame = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Black/White", frame)

    # blur image to help with edge detection
    # 对二值化后的图像 frame 进行高斯模糊处理
    # 高斯模糊是一种平滑图像的滤波技术，它可以有效地减小图像中的随机噪声和椒盐噪声。
    # 在二值化图像中，这些噪声可能表现为孤立的像素或不规则的边缘。
    # 通过高斯模糊，这些噪声点会被周围的像素值平均，从而降低其对后续处理（如边缘检测）的影响。
    # 在水平和垂直方向上使用的高斯滤波器的宽度和高度。较大的内核尺寸会创建更平滑的图像，但也可能导致边缘信息的丢失
    blurred = cv2.GaussianBlur(frame, (9, 9), 0)
    cv2.imshow("Blurred", blurred)

    # identify edges & show on screen
    # 最低阙值：只有超过这个的才被标记为边缘候选者
    # 最高阙值：如果某像素梯度强度超过最低，且其周围元素有足够多的像素超过最高，那么该元素被认定是边缘
    edged = cv2.Canny(blurred, 30, 150)
    cv2.imshow("Edged", edged)

    # perform full Hough Transform to identify lane lines
    # 图像空间中的边缘点转换到参数空间（极坐标表示）的方法
    # 它能够有效地检测出图像中的直线，即使这些直线在原始图像中被部分遮挡或断裂。
    lines = cv2.HoughLines(edged, 1, np.pi / 180, 25)
    print(lines)

    # define arrays for left and right lanes
    rho_left = []
    theta_left = []
    rho_right = []
    theta_right = []

    # ensure cv2.HoughLines found at least one line
    if lines is not None:

        # loop through all of the lines found by cv2.HoughLines
        for i in range(0, len(lines)):

            # evaluate each row of cv2.HoughLines output 'lines'
            for rho, theta in lines[i]:

                # collect left lanes
                if theta < np.pi/2 and theta > np.pi/4:
                    rho_left.append(rho)
                    theta_left.append(theta)

                    # # plot all lane lines for DEMO PURPOSES ONLY
                    # a = np.cos(theta); b = np.sin(theta)
                    # x0 = a * rho; y0 = b * rho
                    # x1 = int(x0 + 400 * (-b)); y1 = int(y0 + 400 * (a))
                    # x2 = int(x0 - 600 * (-b)); y2 = int(y0 - 600 * (a))
                    #
                    # cv2.line(snip, (x1, y1), (x2, y2), (0, 0, 255), 1)

                # collect right lanes
                if theta > np.pi/2 and theta < 3*np.pi/4:
                    rho_right.append(rho)
                    theta_right.append(theta)

                    # # plot all lane lines for DEMO PURPOSES ONLY
                    # a = np.cos(theta); b = np.sin(theta)
                    # x0 = a * rho; y0 = b * rho
                    # x1 = int(x0 + 400 * (-b)); y1 = int(y0 + 400 * (a))
                    # x2 = int(x0 - 600 * (-b)); y2 = int(y0 - 600 * (a))
                    #
                    # cv2.line(snip, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # statistics to identify median lane dimensions
    # 取一个中位数
    left_rho = np.median(rho_left)
    left_theta = np.median(theta_left)
    right_rho = np.median(rho_right)
    right_theta = np.median(theta_right)

    # plot median lane on top of scene snip
    '''
    这是一个条件判断，检查计算出的左侧车道线极角（left_theta）是否大于π/4（即60度）。
    这是因为霍夫变换中的极角范围是-π到π，而我们通常希望在图像中绘制水平或接近水平的车道线。
    如果left_theta大于π/4，说明这条车道线有一定的倾斜，需要进行坐标变换。
    '''
    if left_theta > np.pi/4:
        a = np.cos(left_theta); b = np.sin(left_theta)
        x0 = a * left_rho; y0 = b * left_rho
        # 定义两个偏移量，用于确定要绘制的车道线的长度。
        # offset1是从交点开始向左（负x方向）的距离,offset2是从交点开始向右（正x方向）的距离
        offset1 = 250; offset2 = 800
        x1 = int(x0 - offset1 * (-b)); y1 = int(y0 - offset1 * (a))
        x2 = int(x0 + offset2 * (-b)); y2 = int(y0 + offset2 * (a))
        # 使用OpenCV的line()函数在图像snip上绘制车道线。
        # 输入参数包括：图像、起始坐标、结束坐标、线条颜色（这里是绿色，BGR格式）和线条宽度（这里是6像素）
        cv2.line(snip, (x1, y1), (x2, y2), (0, 255, 0), 6)

    if right_theta > np.pi/4:
        a = np.cos(right_theta); b = np.sin(right_theta)
        x0 = a * right_rho; y0 = b * right_rho
        offset1 = 290; offset2 = 800
        x3 = int(x0 - offset1 * (-b)); y3 = int(y0 - offset1 * (a))
        x4 = int(x0 - offset2 * (-b)); y4 = int(y0 - offset2 * (a))

        cv2.line(snip, (x3, y3), (x4, y4), (255, 0, 0), 6)



    # overlay semi-transparent lane outline on original
    # 在原始图像（snip）上叠加一个半透明的车道线轮廓
    if left_theta > np.pi/4 and right_theta > np.pi/4:
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)

        # (1) create a copy of the original:
        overlay = snip.copy()
        # (2) draw shapes:
        # 在 overlay 图像上绘制一个多边形。输入参数包括：目标图像、顶点坐标数组、填充颜色（这里是绿色，BGR格式）
        cv2.fillConvexPoly(overlay, pts, (0, 255, 0))
        # (3) blend with the original:
        # 使用OpenCV的 addWeighted() 函数将 overlay 图像与原始图像 snip 进行混合。
        # opacity 参数定义了叠加图像的透明度（取值范围为0到1）
        opacity = 0.4
        cv2.addWeighted(overlay, opacity, snip, 1 - opacity, 0, snip)

    cv2.imshow("Lined", snip)


    # perform probablistic Hough Transform to identify lane lines
    # lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 20, 2, 1)
    # for x in range(0, len(lines)):
    #     for x1, y1, x2, y2 in lines[x]:
    #         cv2.line(snip, (x1, y1), (x2, y2), (0, 0, 255), 2)


    # press the q key to break out of video
    # 如果用户在25毫秒内按下了 'q' 键，那么程序就会执行 break 语句，跳出当前循环或者终止程序。
    # 这通常用于在显示图像的同时，提供一种让用户手动停止程序的方式。
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# clear everything once finished
# 确保该资源被正确关闭和释放，防止内存泄漏或者设备被占用
cap.release()
cv2.destroyAllWindows()

