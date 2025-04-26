import cv2
import numpy as np
import os.path
from PIL import Image
from PIL import ImageDraw, ImageFont

color_ranges_hsv = {
    'red': [
        # 红色 (靠近0)
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        # 红色 (靠近179)
        (np.array([160, 100, 100]), np.array([179, 255, 255]))
    ],
    'orange': [
        (np.array([11, 100, 100]), np.array([25, 255, 255]))
    ],
    'yellow': [
        (np.array([26, 100, 100]), np.array([40, 255, 255]))
    ],
    'green': [
        (np.array([41, 50, 50]), np.array([85, 255, 255]))  # 绿色S和V可以稍低
    ],
    'cyan': [  # 青色 (介于绿和蓝之间)
        (np.array([86, 50, 50]), np.array([100, 255, 255]))
    ],
    'blue': [
        (np.array([101, 50, 50]), np.array([125, 255, 255]))
    ],
    'purple': [  # 紫色 (介于蓝和红之间，靠近高H值)
        (np.array([126, 50, 50]), np.array([165, 255, 255]))
    ],
    'black': [
        (np.array([0, 0, 0]), np.array([179, 255, 30]))  # 低V值
    ],
    'white': [
        (np.array([0, 0, 220]), np.array([179, 30, 255]))  # 高V值，低S值
    ],
    'gray': [
        (np.array([0, 0, 31]), np.array([179, 30, 219]))  # 中等V值，低S值 (排除黑和白)
    ]
}

color_ranges_rgb = {
    # Red series
    'deep_red': (np.array([0, 0, 128]), np.array([30, 30, 255])),
    'red': (np.array([0, 0, 200]), np.array([30, 30, 255])),
    'light_red': (np.array([150, 150, 200]), np.array([180, 180, 255])),
    'pink': (np.array([180, 180, 220]), np.array([220, 220, 255])),

    # Blue series
    'deep_blue': (np.array([128, 0, 0]), np.array([255, 30, 30])),
    'blue': (np.array([200, 0, 0]), np.array([255, 30, 30])),
    'sky_blue': (np.array([200, 150, 0]), np.array([255, 200, 50])),
    'light_blue': (np.array([200, 180, 150]), np.array([255, 220, 200])),

    # Green series
    'deep_green': (np.array([0, 128, 0]), np.array([30, 255, 30])),
    'green': (np.array([0, 200, 0]), np.array([30, 255, 30])),
    'light_green': (np.array([150, 200, 150]), np.array([200, 255, 200])),
    'yellow_green': (np.array([100, 200, 100]), np.array([150, 255, 150])),

    # Yellow series
    'golden': (np.array([0, 180, 180]), np.array([30, 255, 255])),
    'yellow': (np.array([0, 200, 200]), np.array([30, 255, 255])),
    'light_yellow': (np.array([150, 200, 200]), np.array([200, 255, 255])),

    # Purple series
    'deep_purple': (np.array([128, 0, 128]), np.array([255, 30, 255])),
    'purple': (np.array([200, 0, 200]), np.array([255, 30, 255])),
    'light_purple': (np.array([200, 150, 200]), np.array([255, 200, 255])),

    # Others
    'brown': (np.array([0, 75, 150]), np.array([30, 120, 200])),
    'orange': (np.array([0, 125, 200]), np.array([30, 175, 255])),
    'white': (np.array([220, 220, 220]), np.array([255, 255, 255])),
    'gray': (np.array([128, 128, 128]), np.array([192, 192, 192])),
    'black': (np.array([0, 0, 0]), np.array([32, 32, 32]))
}


def maskProcess(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def colorCatagory(hsvFrame, originalFrame, offsetX, offsetY):
    if hsvFrame is None:
        print("Error: 输入的 hsvFrame 为 None。")
        return None, originalFrame

    # --- 红色 (需要合并两个范围) ---
    lower_red1, upper_red1 = color_ranges_hsv['red'][0]
    red_mask1 = cv2.inRange(hsvFrame, lower_red1, upper_red1)
    lower_red2, upper_red2 = color_ranges_hsv['red'][1]
    red_mask2 = cv2.inRange(hsvFrame, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # --- 橙色 ---
    lower_orange, upper_orange = color_ranges_hsv['orange'][0]
    orange_mask = cv2.inRange(hsvFrame, lower_orange, upper_orange)

    # --- 黄色 ---
    lower_yellow, upper_yellow = color_ranges_hsv['yellow'][0]
    yellow_mask = cv2.inRange(hsvFrame, lower_yellow, upper_yellow)

    # --- 绿色 ---
    lower_green, upper_green = color_ranges_hsv['green'][0]
    green_mask = cv2.inRange(hsvFrame, lower_green, upper_green)

    # --- 青色 ---
    lower_cyan, upper_cyan = color_ranges_hsv['cyan'][0]
    cyan_mask = cv2.inRange(hsvFrame, lower_cyan, upper_cyan)

    # --- 蓝色 ---
    lower_blue, upper_blue = color_ranges_hsv['blue'][0]
    blue_mask = cv2.inRange(hsvFrame, lower_blue, upper_blue)

    # --- 紫色 ---
    lower_purple, upper_purple = color_ranges_hsv['purple'][0]
    purple_mask = cv2.inRange(hsvFrame, lower_purple, upper_purple)

    # --- 黑色 ---
    lower_black, upper_black = color_ranges_hsv['black'][0]
    black_mask = cv2.inRange(hsvFrame, lower_black, upper_black)

    # --- 白色 ---
    lower_white, upper_white = color_ranges_hsv['white'][0]
    white_mask = cv2.inRange(hsvFrame, lower_white, upper_white)

    # --- 灰色 ---
    lower_gray, upper_gray = color_ranges_hsv['gray'][0]
    gray_mask = cv2.inRange(hsvFrame, lower_gray, upper_gray)

    # 掩膜处理
    red_mask = maskProcess(red_mask)
    orange_mask = maskProcess(orange_mask)
    yellow_mask = maskProcess(yellow_mask)
    green_mask = maskProcess(green_mask)
    cyan_mask = maskProcess(cyan_mask)
    blue_mask = maskProcess(blue_mask)
    purple_mask = maskProcess(purple_mask)
    black_mask = maskProcess(black_mask)
    white_mask = maskProcess(white_mask)
    gray_mask = maskProcess(gray_mask)

    total_mask = red_mask + orange_mask + yellow_mask + green_mask + cyan_mask + blue_mask + purple_mask + black_mask + white_mask + gray_mask
    contours, _ = cv2.findContours(total_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_color = None

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # 设置最小区域面积以排除噪声
            x, y, w, h = cv2.boundingRect(contour)
            # 将坐标转换到原始帧的坐标系统
            x_orig = x + offsetX
            y_orig = y + offsetY

            if np.any(red_mask[y:y + h, x:x + w]):
                detected_color = "red"
            elif np.any(blue_mask[y:y + h, x:x + w]):
                detected_color = "blue"
            elif np.any(green_mask[y:y + h, x:x + w]):
                detected_color = "green"
            elif np.any(yellow_mask[y:y + h, x:x + w]):
                detected_color = "yellow"
            elif np.any(orange_mask[y:y + h, x:x + w]):
                detected_color = "orange"
            elif np.any(purple_mask[y:y + h, x:x + w]):
                detected_color = "purple"
            elif np.any(cyan_mask[y:y + h, x:x + w]):
                detected_color = "cyan"
            elif np.any(black_mask[y:y + h, x:x + w]):
                detected_color = "black"
            elif np.any(white_mask[y:y + h, x:x + w]):
                detected_color = "white"
            elif np.any(gray_mask[y:y + h, x:x + w]):
                detected_color = "gray"

            # 在原始帧上绘制矩形和文本
            cv2.rectangle(originalFrame, (x_orig, y_orig), (x_orig + w, y_orig + h), (255, 255, 255), 4)
            if detected_color:
                cv2.putText(originalFrame, detected_color, (x_orig, y_orig - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 5)

    return detected_color, originalFrame


def centerFrame(frame):
    # 获取中心区域及其偏移量
    height, width = frame.shape[:2]
    center_width = width // 2
    center_height = height // 2
    offset_x = (width - center_width) // 2
    offset_y = (height - center_height) // 2

    # 绘制中心区域矩形
    cv2.rectangle(frame, (offset_x, offset_y),
                  (offset_x + center_width, offset_y + center_height),
                  (255, 0, 0), 2)

    # 提取中心区域并进行颜色检测
    center_frame = frame[offset_y:offset_y + center_height,
                   offset_x:offset_x + center_width]
    return center_frame, offset_x, offset_y


def apply_gray_world(bgr_image):
    """
    应用灰度世界算法进行颜色恒常性处理，以减少光照颜色对结果的影响。

    Args:
        bgr_image: 输入的BGR格式图像 (NumPy数组)。

    Returns:
        颜色校正后的BGR格式图像 (NumPy数组, uint8)。
        如果校正失败（例如平均值为零），则返回原始图像。
    """
    # 检查输入图像是否有效
    if bgr_image is None or bgr_image.size == 0:
        print("警告: apply_gray_world 收到 None 或空图像。")
        return bgr_image

    # 转换为float32进行计算，避免溢出/下溢
    img_float = bgr_image.astype(np.float32)
    # 再次检查转换后的图像大小
    if img_float.size == 0:
        return bgr_image

    # 分别计算B, G, R通道的平均值
    avg_b = np.mean(img_float[:, :, 0])
    avg_g = np.mean(img_float[:, :, 1])
    avg_r = np.mean(img_float[:, :, 2])

    # 防止通道平均值为零导致除零错误
    if avg_b == 0 or avg_g == 0 or avg_r == 0:
        print("警告: 一个或多个通道的平均值为零。跳过灰度世界算法。")
        return bgr_image # 如果无法进行校正，返回原始图像

    # 计算所有通道的整体平均灰度值
    avg_gray = (avg_b + avg_g + avg_r) / 3.0

    # 计算每个通道的缩放因子，使得校正后各通道平均值趋向于avg_gray
    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r

    # 将缩放因子应用于每个通道
    img_float[:, :, 0] *= scale_b
    img_float[:, :, 1] *= scale_g
    img_float[:, :, 2] *= scale_r

    # 将像素值裁剪到有效的[0, 255]范围
    corrected_img = np.clip(img_float, 0, 255)

    # 转换回uint8格式
    corrected_img_uint8 = corrected_img.astype(np.uint8)

    return corrected_img_uint8


def centerHSVAnalysis(hsvFrame):
    # 检查 hsvFrame 是否为 None
    if hsvFrame is None:
        print("Error: 输入的 hsvFrame 为 None。")
        return None

    # 重塑图像数组为二维数组
    pixels = hsvFrame.reshape(-1, 3)
    pixels = np.float32(pixels)

    # 定义K-means参数
    k = 3  # 提取3个主要颜色
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.5)

    # 执行K-means聚类
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 计算每个聚类的像素数量
    _, counts = np.unique(labels, return_counts=True)

    # 将主导颜色按占比排序
    sorted_indices = np.argsort(counts)[::-1]
    sorted_centers = centers[sorted_indices]

    # 返回主导颜色的HSV值
    dominant_color = sorted_centers[0]
    return dominant_color


def generateCollection(dominant_color):
    """
    根据主导颜色生成多种配色方案
    参数: dominant_color - HSV格式的主导颜色 [H, S, V]
    返回: 包含不同配色方案的字典
    """
    # 获取主色调H值(0-179)
    h = dominant_color[0]
    s = dominant_color[1]
    v = dominant_color[2]

    # 创建配色方案字典
    color_schemes = {
        'complementary': [],  # 互补色
        'analogous': [],  # 邻近色
        'triadic': [],  # 三色
        'split_complementary': []  # 分裂互补色
    }

    # 1. 互补色 (相差180度)
    complement_h = (h + 90) % 180  # OpenCV中H范围是0-179，相当于色环360度的一半
    color_schemes['complementary'] = [
        [h, s, v],
        [complement_h, s, v]
    ]

    # 2. 邻近色 (相差30度)
    analogous_h1 = (h + 15) % 180
    analogous_h2 = (h - 15) % 180
    color_schemes['analogous'] = [
        [h, s, v],
        [analogous_h1, s, v],
        [analogous_h2, s, v]
    ]

    # 3. 三色配色 (相差120度)
    triadic_h1 = (h + 60) % 180
    triadic_h2 = (h + 120) % 180
    color_schemes['triadic'] = [
        [h, s, v],
        [triadic_h1, s, v],
        [triadic_h2, s, v]
    ]

    # 4. 分裂互补色 (互补色两侧各30度)
    split_h1 = (complement_h + 15) % 180
    split_h2 = (complement_h - 15) % 180
    color_schemes['split_complementary'] = [
        [h, s, v],
        [split_h1, s, v],
        [split_h2, s, v]
    ]

    # 转换所有HSV值为整数
    for scheme in color_schemes:
        color_schemes[scheme] = [[int(c) for c in color] for color in color_schemes[scheme]]

    return color_schemes


def hsvToName(hsv_scheme):
    """
    将HSV颜色值转换为对应的颜色名称
    参数: hsv_scheme - HSV颜色值列表 [[H,S,V], ...]
    返回: 颜色名称列表
    """
    result = []
    for hsv in hsv_scheme:
        h, s, v = hsv

        # 根据饱和度和亮度先判断是否是黑白灰
        if v < 30:
            result.append('black')
            continue
        elif v > 220 and s < 30:
            result.append('white')
            continue
        elif s < 30:
            result.append('gray')
            continue

        # 根据色相判断颜色
        if 0 <= h < 10 or 160 <= h <= 179:
            if v > 200 and s > 200:
                result.append('light_red')
            elif v > 150:
                result.append('red')
            else:
                result.append('deep_red')

        elif 11 <= h <= 25:
            result.append('orange')

        elif 26 <= h <= 40:
            if v > 200:
                result.append('light_yellow')
            else:
                result.append('yellow')

        elif 41 <= h <= 85:
            if v > 200:
                result.append('light_green')
            elif v > 150:
                result.append('green')
            else:
                result.append('deep_green')

        elif 86 <= h <= 100:
            result.append('cyan')

        elif 101 <= h <= 125:
            if v > 200:
                result.append('light_blue')
            elif v > 150:
                result.append('blue')
            else:
                result.append('deep_blue')

        elif 126 <= h <= 155:
            if v > 200:
                result.append('light_purple')
            elif v > 150:
                result.append('purple')
            else:
                result.append('deep_purple')
        else:
            result.append('unknown')

    return result

def centerProcess(center_frame):
    # 调整对比度和亮度
    alpha = 1.3  # 对比度因子
    beta = 0  # 亮度偏移
    center_frame = cv2.convertScaleAbs(center_frame, alpha=alpha, beta=beta)

    # 自适应直方图均衡化
    lab = cv2.cvtColor(center_frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    center_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 中值滤波去除椒盐噪声
    center_frame = cv2.medianBlur(center_frame, 3)

    # --- 应用颜色补偿 (灰度世界算法) ---
    # 对输入的ROI进行颜色校正，减少光照影响
    center_frame_GW = apply_gray_world(center_frame)
    return center_frame_GW




def create_color_blocks(color_schemes, index, block_size=(100, 50)):
    """
    在一张图片中展示所有配色方案
    :param color_schemes: HSV配色方案字典
    :param block_size: 色块大小 (宽度, 高度)
    :return: 包含所有配色方案的图像
    """
    # 计算总高度 (每个方案占用block_size[1]*2高度，包含文字区域)
    total_height = len(color_schemes) * (block_size[1] * 2)
    # 找出最长的配色方案，确定总宽度
    max_colors = max(len(colors) for colors in color_schemes.values())
    total_width = max_colors * block_size[0]

    # 创建背景图像（白色）
    background = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(background)

    # 尝试加载字体（如果没有则使用默认字体）
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # 为每个配色方案创建色块
    y_offset = 0
    for scheme_name, colors in color_schemes.items():
        # 添加方案名称
        draw.text((10, y_offset + 10), scheme_name, font=font, fill=(0, 0, 0))

        # 在标签下方创建色块
        y_pos = y_offset + block_size[1]
        for i, hsv in enumerate(colors):
            # 创建单色HSV图像
            hsv_block = np.full((1, 1, 3), hsv, dtype=np.uint8)
            # 转换为BGR
            bgr_block = cv2.cvtColor(hsv_block, cv2.COLOR_HSV2BGR)
            # 转换为RGB
            rgb_block = (int(bgr_block[0, 0, 2]), int(bgr_block[0, 0, 1]), int(bgr_block[0, 0, 0]))

            # 创建单色块
            color_block = Image.new('RGB', block_size, rgb_block)
            # 粘贴到主图像
            background.paste(color_block, (i * block_size[0], y_pos))

        # 更新y偏移量，为下一个方案预留空间
        y_offset += block_size[1] * 2

    # 保存图像
    save_dir = './color_schemes'
    os.makedirs(save_dir, exist_ok=True)
    background.save(os.path.join(save_dir, f'combined_schemes_{index}.png'))