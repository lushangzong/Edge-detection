###########################################################
#作者：鲁尚宗 时间2018年 12月4日
#本程序实现了不同计算方式和不同边缘检测算子的卷积
#两种计算方式：普通循环计算、快速傅里叶变速
#算子：sobel算子、prewitt算子、拉普拉斯算子
#结果发现傅里叶变换加速的方法比普通方法至少快100倍，结果一致
#在未设置阈值的情况下，sobel算子提取的边缘比较平滑
# prewitt算子结果和sobel算子相差不大
#拉普拉斯算子结果噪声较多
############################################################
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import time


# ##################################################
# 绘制图像数组的函数，传入一个图像数组，绘制它的图像
# array为实数数组，name为图片的名字
# ##################################################
def draw(array, name):
    plt.imshow(array, plt.cm.gray)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title(name, fontproperties='SimHei')  # 原始图像 图像题目
    plt.show()


########################################################
# 普通卷积操作的函数
# 传入图像矩阵和卷积核，采用最普通的循环来实现卷积
# 返回卷积结果取值范围为0~255
# 裁剪掉边缘，所以返回结果会比原图像小一圈
########################################################
def conv(array, kernel):
    t0 = time.process_time()

    # 先将卷积核旋转180度
    kernel = np.flipud(kernel)  # 上下翻转
    kernel = np.fliplr(kernel)  # 左右翻转

    # 获取原来图像的大小
    array_height = array.shape[0]
    array_width = array.shape[1]

    # 获取核的大小
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    # 结果图像的大小，裁剪掉了一圈
    new_height = array_height - kernel_height + 1
    new_width = array_width - kernel_width + 1

    # 用来储存结果的矩阵，为了精度先定义类型为浮点型
    new_img = np.zeros([new_height, new_width], np.float64)

    # 为了对应像元相乘的偏移量
    h_offset = kernel_height // 2
    w_offset = kernel_width // 2

    # 因为小了一圈所以不从0开始
    for i in range(h_offset, new_height - h_offset):
        for j in range(w_offset, new_width - w_offset):
            value = 0
            # 对应像元相乘
            for m in range(kernel_height):
                for n in range(kernel_width):
                    # 通过偏移量来确定是哪个像元
                    value = value + array[i - h_offset + m, j - w_offset + n] * kernel[m, n]
            new_img[i - h_offset, j - w_offset] = value

    # 结果中有正有负，将其转化为0~255
    result = np.uint8(np.absolute(new_img))

    print('普通卷积方式的时间：', time.process_time() - t0, "seconds time")
    return result


##################################################
# 通过快速傅里叶变换实现卷积加速
# 传入图像矩阵和卷积核，采用最普通的循环来实现卷积
# 返回卷积结果取值范围为0~255
# 返回结果并不会小
##################################################
def fft_conv(array, kernel):
    t0 = time.process_time()

    # 获取原图像信息
    array_height = array.shape[0]
    array_width = array.shape[1]

    # 将图像和核都进行快速傅里叶变换，
    array_fft = np.fft.fft2(array)
    # 为了相乘规定了结果的大小和原图一样
    kernel_fft = np.fft.fft2(kernel, [array_height, array_width])

    # 二者对应像元相乘
    fft_multi = array_fft * kernel_fft
    # 取实数部分
    new_img = np.fft.ifft2(fft_multi).real

    # 将范围变为0~255
    result = np.uint8(np.absolute(new_img))

    print('快速傅里叶变换卷积方式的时间：', time.process_time() - t0, "seconds time")
    return result


# 打开黑白图像
img = Image.open("horse.jpeg")

# 将图像转换为数组，截取其中一个波段
img_array = np.array(img)
img1 = img_array[:, :, 0]

#sobel 算子
sobel = np.array(([-1, -1, 0], [-1, 0, 1], [0, 1, 1]))

# prewitt算子
prewitt = np.array(([-2, -1, 0], [-1, 0, 1], [0, 1, 2]))

# 拉普拉斯算子1
laplacian_1 = np.array(([0, -1, 0], [-1, 4, -1], [0, -1, 0]))

#拉普拉斯算子2
laplacian_2 = np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]))

print('soble算子')
img_soble1 = conv(img1, sobel)
draw(img_soble1,'sobel算子 普通计算')
img_sobel2=fft_conv(img1,sobel)
draw(img_sobel2,'sobel算子 快速傅里叶加速')
print('\n')

print('prewitt算子')
img_prewitt1 = conv(img1, prewitt)
draw(img_prewitt1,'prewitt算子 普通计算')
img_prewitt2=fft_conv(img1,prewitt)
draw(img_prewitt2,'prewitt算子 快速傅里叶加速')
print('\n')

print('拉普拉斯算子1')
img_laplacian_11 = conv(img1, laplacian_1)
draw(img_laplacian_11,'拉普拉斯算子1 普通计算')
img_laplacian_12=fft_conv(img1,laplacian_1)
draw(img_laplacian_12,'拉普拉斯算子1 快速傅里叶加速')
print('\n')

print('拉普拉斯算子2')
img_laplacian_21 = conv(img1, laplacian_2)
draw(img_laplacian_21,'拉普拉斯算子2 普通计算')
img_laplacian_22=fft_conv(img1,laplacian_2)
draw(img_laplacian_22,'拉普拉斯算子2 快速傅里叶加速')







