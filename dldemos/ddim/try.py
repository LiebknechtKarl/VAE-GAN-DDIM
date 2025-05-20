import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
# image_path = '/home/ubuntu/Desktop/rl/learn_rl_simple/model_new/net_01_defussion/DL-Demos/work_dirs/diffusion_ddim_sigma_hat/0.jpg'
# /media/ubuntu/1T1/ubuntu_data/work/demo_03_armencode/编码点重命名/0000_2.BMP
# /home/ubuntu/Pictures/Screenshot from 2024-08-10 18-06-29.png
# image_path = '/home/ubuntu/Pictures/Screenshot from 2024-08-10 18-06-29.png'

image_path = '/home/ubuntu/Pictures/Screenshot from 2024-08-16 11-33-04.png'


import cv2
import numpy as np

def apply_lighting(image, center=None, intensity=1.0, radius=None):
    """
    在图像上应用光照效果。
    
    参数:
    - image: 输入图像 (H, W, 3)。
    - center: 光源的中心位置 (x, y)。默认为图像中心。
    - intensity: 光照强度。默认为1.0。
    - radius: 光照影响的半径。默认为图像的对角线长度。
    
    返回:
    - 加光后的图像。
    """
    # 获取图像的维度
    h, w = image.shape[:3]
    
    # 默认光源位置为图像中心
    if center is None:
        center = (w // 2, h // 2)
    
    # 默认光照半径为图像对角线长度
    if radius is None:
        radius = np.sqrt((w // 2) ** 2 + (h // 2) ** 2)
    
    # 创建光照掩码
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, center, int(radius), (255 * intensity), thickness=-1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius / 2, sigmaY=radius / 2)
    
    # 将光照掩码应用到图像
    for i in range(3):  # 对每个颜色通道
        image[:, :, i] = cv2.add(image[:, :, i], mask.astype(np.uint8))
    
    return image




# 示例使用
# image_path = 'your_image_path_here'
image = cv2.imread(image_path)

print(image)


# 应用光照效果
light_image = apply_lighting(image, center=(300, 300), intensity=1.5, radius=300)

# 显示原始图像和加光后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Image with Lighting', light_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

