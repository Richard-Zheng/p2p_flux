import os
from PIL import Image, ImageDraw

def create_visual_icl_grid(a_path, aa_path, b_path, output_grid_path="grid.png", output_mask_path="mask.png", sub_img_size=512):
    """
    拼接 Visual In-Context Learning 所需的四方格图像并生成 Mask。
    
    参数:
        a_path (str): 示例原图 A 的路径
        aa_path (str): 示例结果 A' 的路径
        b_path (str): 目标原图 B 的路径
        output_grid_path (str): 输出的拼接图像路径
        output_mask_path (str): 输出的 Mask 图像路径
        sub_img_size (int): 每个子图的宽高尺寸 (默认512，则总图为 1024x1024)
    """
    
    # 1. 检查文件是否存在
    for path in [a_path, aa_path, b_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到图像文件: {path}")

    # 2. 读取并统一缩放图像
    # 将图像转换为 RGB 模式，避免 RGBA 通道带来的拼接问题
    img_a = Image.open(a_path).convert("RGB").resize((sub_img_size, sub_img_size))
    img_aa = Image.open(aa_path).convert("RGB").resize((sub_img_size, sub_img_size))
    img_b = Image.open(b_path).convert("RGB").resize((sub_img_size, sub_img_size))

    # 3. 创建 2x2 网格画布 (总尺寸为 2倍的 sub_img_size)
    grid_size = sub_img_size * 2
    # 默认背景为中性灰色 (128,128,128)，右下角的区域在重绘前会显示为灰色
    grid_img = Image.new("RGB", (grid_size, grid_size), color=(128, 128, 128))

    # 4. 拼接图像到指定位置
    # 左上角 (0, 0)
    grid_img.paste(img_a, (0, 0))
    # 右上角 (sub_img_size, 0)
    grid_img.paste(img_aa, (sub_img_size, 0))
    # 左下角 (0, sub_img_size)
    grid_img.paste(img_b, (0, sub_img_size))
    # 右下角 (sub_img_size, sub_img_size) 留空，等待模型 Inpainting

    # 5. 保存拼接好的网格图
    grid_img.save(output_grid_path)
    print(f"✅ 网格图像已保存至: {output_grid_path}")

    # 6. 生成 Mask 图像
    # 创建纯黑色的灰度图 ("L" 模式)
    mask_img = Image.new("L", (grid_size, grid_size), color=0)
    draw = ImageDraw.Draw(mask_img)
    
    # 在右下角绘制白色矩形 (255 表示需要 Inpainting 的区域)
    # 坐标格式: [左上角x, 左上角y, 右下角x, 右下角y]
    draw.rectangle(
        [sub_img_size, sub_img_size, grid_size, grid_size], 
        fill=255
    )

    # 7. 保存 Mask 图
    mask_img.save(output_mask_path)
    print(f"✅ Mask 图像已保存至: {output_mask_path}")

if __name__ == "__main__":
    # 假设你的三张图片和脚本在同一个目录下
    # 你可以根据需要修改 sub_img_size，比如对于 Stable Diffusion XL，常常设置为 512 (总图1024x1024)
    create_visual_icl_grid(
        a_path="a.png",
        aa_path="aa.png",
        b_path="b.png",
        output_grid_path="grid.png",
        output_mask_path="mask.png",
        sub_img_size=512 
    )