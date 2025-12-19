# local_inference_server.py (已添加简洁的日志文件记录和图像内统计显示功能)
print("!!!!!!!!!! 正在运行的是最新修改过的版本 (带文件日志和图像内统计) !!!!!!!!!!")
import cv2
import numpy as np
import time
import torch
from sklearn.cluster import KMeans
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import asyncio
import websockets
import json
import base64
import logging # <<< 1. 导入 logging 模块

# ==============================================================================
#                                配置参数
# ==============================================================================

# --- WebSocket 服务器配置 ---
HOST, PORT = "0.0.0.0", 8765 # 服务器监听的 IP 地址和端口。"0.0.0.0" 表示监听所有可用网络接口。
# --- 新增：服务器输出目录配置 ---
SERVER_OUTPUT_DIR = "./output" # <--- 新增：定义一个用于保存结果图片的目录

# --- Segment Anything Model (SAM) 模型配置 ---
SAM_CHECKPOINT = 'sam_vit_b_01ec64.pth' # SAM 预训练模型的权重文件路径。
SAM_MODEL_TYPE = "vit_b"                        # SAM 模型的类型，"vit_b" 代表基础版的 Vision Transformer。必须与权重文件匹配。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # 推理设备。如果检测到可用的 NVIDIA GPU，则使用 "cuda"，否则使用 "cpu"。

# --- 图像处理与几何计算参数 ---
MAX_IMAGE_SIZE = 1024               # 图像预处理时的最大尺寸。将图像等比缩放到此尺寸以内，以加快 SAM 处理速度并节省显存。
KNOWN_DIAMETERS_MM = [6, 8, 10, 12, 16, 20, 25] # 预先知道的球体真实直径（毫米）。用于 K-Means 聚类后的标签映射。
N_CLUSTERS = len(KNOWN_DIAMETERS_MM)  # K-Means 聚类的簇数，直接由已知的直径种类数确定。

# --- SAM 自动掩码生成器 (SamAutomaticMaskGenerator) 参数 ---
# 这些参数精细地控制 SAM 如何在没有特定提示的情况下自动分割图像中的所有对象。
SAM_GEN_CFG = {
    "points_per_side": 36,          # 在图像网格的每一边上采样的点数。点数越多，可能检测到越多的细节，但速度越慢。32 是一个平衡点。
    "pred_iou_thresh": 0.90,        # 预测的 IoU (交并比) 阈值。只有当模型预测其生成的掩码质量（IoU）高于此值时，才会保留该掩码。
    "stability_score_thresh": 0.95, # 稳定性得分阈值。衡量当输入提示点轻微变动时，掩码的变化程度。高分表示掩码非常稳定和可靠。
    "crop_n_layers": 1,             # 图像裁剪的层数。设置为 1 会对全图进行一次额外的裁剪和处理，有助于发现更小的物体。
    "min_mask_region_area": 90      # 掩码的最小有效区域面积（像素）。用于过滤掉 SAM 内部后处理后产生的非常小的、可能是噪声的区域。
}

# --- 后处理与筛选参数 ---
MIN_AREA = 90                       # 轮廓的最小面积（像素）。在找到所有轮廓后，面积小于此值的轮廓将被直接过滤掉。
CIRCULARITY_THRESHOLD = 0.7         # 圆度阈值。圆度计算公式为 (4 * pi * 面积) / (周长^2)，理想圆为 1。低于此值的轮廓被认为不够圆，将被过滤。
FILTER_BORDER_OBJECTS = True        # 是否过滤掉接触图像边缘的对象。True 表示接触边缘的对象会被丢弃，因为它们可能不完整。
OVERLAP_THRESHOLD = 0.4             # 重叠阈值。用于去除重复检测。如果两个圆心的距离小于它们最大半径的 0.4 倍，则认为它们是重复的，只保留一个。

# --- 分水岭算法参数 ---
SPLIT_AREA_THRESHOLD = 500          # 掩码面积阈值（像素）。当一个掩码的面积大于此值时，会尝试使用分水岭算法将其分割成多个独立的实例。
DIST_TRANS_REL = 0.2                # 距离变换的相对阈值。在分水岭算法中，用于确定“确定前景”区域。值为 0.2 表示距离变换结果中大于最大距离 20% 的区域被视为前景核心。


# <<< ==================== 2. 新增：日志配置函数 ====================
def setup_logging():
    """配置日志记录器，将简洁的结果写入文件"""
    # 获取名为 'detection_results' 的记录器实例
    logger = logging.getLogger('detection_results')
    logger.setLevel(logging.INFO) # 设置此记录器处理的最低日志级别为 INFO

    # 如果记录器已经有了 handlers (例如在某些交互式环境中重复运行时)，先清空，防止日志重复记录
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建一个文件 handler，用于将日志写入文件。
    # 模式为 'a' (append)，表示在文件末尾追加日志，而不是覆盖。
    # encoding='utf-8' 确保中文字符能正确写入。
    file_handler = logging.FileHandler('detection_log.txt', mode='a', encoding='utf-8')
    
    # 创建一个日志格式化器，定义日志的输出格式。
    # %(asctime)s: 日志记录的时间
    # %(message)s: 日志消息本身
    # datefmt: 定义时间的显示格式
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    # 将配置好的文件 handler 添加到记录器中
    logger.addHandler(file_handler)

    return logger

# 在程序启动时获取日志记录器实例，供后续使用
result_logger = setup_logging()
# <<< =============================================================

def enhance_image(image):
    """
    对输入图像进行增强，以提高 SAM 的分割效果。
    - 转换到 LAB 色彩空间，只对亮度(L)通道进行操作，避免影响颜色。
    - 使用 CLAHE (对比度受限的自适应直方图均衡化) 提升局部对比度。
    - 使用一个简单的拉普拉斯核进行锐化，使边缘更清晰。
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB); l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)); cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b)); enhanced_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(src=enhanced_clahe, ddepth=-1, kernel=kernel)
    return sharpened_image

def split_mask_watershed(mask_bin):
    """
    使用分水岭算法尝试分割可能粘连在一起的对象的二值掩码。
    - 首先进行开运算去除小的噪声点。
    - 计算距离变换，找到离背景最远的核心区域（确定前景）。
    - 膨胀掩码得到确定背景。
    - 在确定前景和确定背景之间的未知区域进行分水岭变换，从而找到分割线。
    - 返回分割后的所有独立掩码列表。
    """
    m = (mask_bin > 0).astype('uint8'); kernel = np.ones((3,3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    if dist.max() <= 0: return [ (m * 255).astype('uint8') ]
    ret, sure_fg = cv2.threshold(dist, DIST_TRANS_REL * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg); sure_bg = cv2.dilate(m, kernel, iterations=3).astype('uint8')
    unknown = cv2.subtract(sure_bg, sure_fg); ret2, markers = cv2.connectedComponents(sure_fg)
    if markers.max() == 0: return [ (m * 255).astype('uint8') ]
    markers = markers + 1; markers[unknown == 1] = 0
    fake_color = cv2.cvtColor(m * 255, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(fake_color, markers); res_masks = []
    for lab in range(2, markers.max() + 1):
        part = (markers == lab).astype('uint8') * 255
        if part.sum() > 0: res_masks.append(part)
    if not res_masks: return [ (m * 255).astype('uint8') ]
    return res_masks

def compute_geoms_from_masks(masks, scale_factor, img_h, img_w):
    """
    从掩码列表中计算几何属性，并根据预设规则进行过滤。
    - 对每个掩码，找到其最大轮廓。
    - 过滤掉面积过小、不圆、接触图像边界的对象。
    - 计算对象的最小外接圆，得到中心点和半径。
    - 对所有候选对象进行非极大值抑制（NMS）的简化版，去除重叠的检测结果。
    - 返回最终筛选后的对象列表，每个对象包含其几何信息。
    """
    candidates = []; filtered_stats = {'area': 0, 'circularity': 0, 'border': 0, 'overlap': 0}
    for m in masks:
        m_uint8 = (m > 0).astype('uint8'); cnts, _ = cv2.findContours(m_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        c = max(cnts, key=cv2.contourArea); area = cv2.contourArea(c)
        if area < MIN_AREA: filtered_stats['area'] += 1; continue
        if FILTER_BORDER_OBJECTS:
            x, y, w, h = cv2.boundingRect(c)
            if x <= 1 or y <= 1 or (x + w) >= img_w - 1 or (y + h) >= img_h - 1: filtered_stats['border'] += 1; continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0: continue
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        if circularity < CIRCULARITY_THRESHOLD: filtered_stats['circularity'] += 1; continue
        (cx_px, cy_px), r_px = cv2.minEnclosingCircle(c)
        candidates.append({'cx': cx_px / scale_factor, 'cy': cy_px / scale_factor, 'r': r_px / scale_factor, 'diam': 2.0 * r_px / scale_factor, 'area': area, 'mask': m, 'cx_px': cx_px, 'cy_px': cy_px, 'r_px': r_px})
    candidates.sort(key=lambda x: x['area'], reverse=True)
    final_insts = []
    for cand in candidates:
        is_duplicate = False
        for kept in final_insts:
            dist = np.sqrt((cand['cx_px'] - kept['cx_px'])**2 + (cand['cy_px'] - kept['cy_px'])**2)
            max_r = max(cand['r_px'], kept['r_px'])
            if dist < max_r * OVERLAP_THRESHOLD: is_duplicate = True; break
        if is_duplicate: filtered_stats['overlap'] += 1
        else: final_insts.append(cand)
    print("\n--- 过滤统计 ---"); print(f"  - 面积过小丢弃: {filtered_stats['area']} 个, 非圆形丢弃: {filtered_stats['circularity']} 个")
    if FILTER_BORDER_OBJECTS: print(f"  - 接触边缘丢弃: {filtered_stats['border']} 个")
    print(f"  - 重叠重复丢弃: {filtered_stats['overlap']} 个"); return final_insts

def cluster_and_label(insts):
    """
    使用 K-Means 算法对检测到的对象的像素直径进行聚类，并将聚类结果映射到已知的真实直径。
    - 将所有对象的像素直径作为特征进行 K-Means 聚类，聚类数 N_CLUSTERS。
    - 对聚类中心进行排序，这样最小的像素直径簇总是对应最小的真实直径。
    - 创建一个从旧标签到新排序标签的映射。
    - 为每个对象实例分配新的、有序的类别标签。
    - 返回带标签的对象列表和排序后的聚类中心。
    """
    if len(insts) < N_CLUSTERS:
        print(f"警告: 实例数({len(insts)})少于类别数({N_CLUSTERS})。")
        if not insts: return [], []
        sorted_insts = sorted(insts, key=lambda x: x['diam'])
        for i, inst in enumerate(sorted_insts): inst['label'] = i % N_CLUSTERS
        return sorted_insts, []
    diams = np.array([it['diam'] for it in insts]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init='auto').fit(diams)
    labels, centers = kmeans.labels_, kmeans.cluster_centers_.flatten()
    order = np.argsort(centers); label_map = {old_label: new_label for new_label, old_label in enumerate(order)}
    for i, inst in enumerate(insts): inst['label'] = label_map[labels[i]]
    return insts, centers[order]

# ==================== 核心修改点 1 (开始) ====================
# 修改 visualize 函数，增加 counts 和 total_count 两个参数
def visualize(original_image, labeled_instances, known_diameters, counts, total_count):
    """
    在原始图像上可视化检测和分类结果，并在左上角显示统计信息。
    - 创建一个彩色覆盖层，为每个类别分配一种不同的颜色。
    - 将每个实例的掩码绘制到覆盖层上。
    - 将覆盖层与原始图像半透明地叠加。
    - 在每个实例的中心位置标注其分类后的真实直径。
    - 在图像左上角添加一个半透明背景，并显示总数和各种尺寸球体的数量。
    - 返回最终的可视化图像。
    """
    output_image = original_image.copy()
    overlay = np.zeros_like(output_image, dtype=np.uint8)
    h_orig, w_orig, _ = original_image.shape
    colors = [tuple(map(int, cv2.cvtColor(np.uint8([[[i * (180 / N_CLUSTERS), 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])) for i in range(N_CLUSTERS)]
    
    # 绘制掩码
    for inst in labeled_instances:
        if 'label' not in inst: continue
        label = inst['label']
        color = colors[min(label, len(colors)-1)]
        mask_original_size = cv2.resize(inst['mask'], (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        overlay[mask_original_size > 0] = color
    
    cv2.addWeighted(overlay, 0.5, output_image, 0.5, 0, output_image)
    
    # 绘制每个球体的标签
    for inst in labeled_instances:
        if 'label' not in inst or inst['label'] >= len(known_diameters): continue
        text = f"{known_diameters[inst['label']]}mm"
        cx, cy = int(inst['cx']), int(inst['cy'])
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(output_image, text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3) # 白色描边
        cv2.putText(output_image, text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)       # 黑色文字

    # --- 新增功能：在左上角绘制统计信息 ---
    # 1. 准备要显示的文本行
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (255, 255, 255) # 白色文字
    line_height = 30 # 每行文字的大致高度

    stats_text = [f"Total Count: {total_count}"]
    # 按尺寸（字典的键转为整数）进行排序，以确保显示顺序一致
    for size, count in sorted(counts.items(), key=lambda item: int(item[0])):
        stats_text.append(f"{size}mm: {count}")

    # 2. 计算文本区域的大小，以便绘制背景
    max_text_width = 0
    for text in stats_text:
        (text_width, _), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        if text_width > max_text_width:
            max_text_width = text_width
            
    margin = 15 # 边距
    rect_w = max_text_width + 2 * 10 # 矩形宽度 = 最大文本宽度 + 左右边距
    rect_h = len(stats_text) * line_height + 10 # 矩形高度 = 行数 * 行高 + 上下边距

    # 3. 绘制半透明背景
    # 定义背景区域的 ROI (Region of Interest)
    roi = output_image[margin:margin+rect_h, margin:margin+rect_w]
    # 创建一个黑色的矩形
    black_rect = np.zeros(roi.shape, dtype=np.uint8)
    # 将黑色矩形和 ROI 区域进行加权混合，实现半透明效果
    res = cv2.addWeighted(roi, 0.5, black_rect, 0.5, 1.0)
    # 将混合后的结果放回原图
    output_image[margin:margin+rect_h, margin:margin+rect_w] = res

    # 4. 在背景上逐行写字
    y_pos = margin + line_height - 5 # 第一行文字的 y 坐标
    for text in stats_text:
        cv2.putText(output_image, text, (margin + 10, y_pos), font, font_scale, text_color, font_thickness)
        y_pos += line_height # 移动到下一行
    # --- 新增功能结束 ---
    
    return output_image
# ==================== 核心修改点 1 (结束) ====================

def process_frame(img, mask_generator):
    """
    处理单帧图像的完整流程。
    1. 图像增强。
    2. 图像缩放以适应模型。
    3. 调用 SAM 模型生成所有对象的掩码。
    4. 对过大的掩码使用分水岭算法进行分割。
    5. 从掩码计算几何属性并进行过滤。
    6. 对合格的对象进行聚类和标记。
    7. 统计每种直径的数量。
    8. 生成可视化结果图。
    """
    print("正在对接收到的图像进行增强处理...")
    enhanced_img = enhance_image(img)
    h, w, _ = enhanced_img.shape
    scale_factor = MAX_IMAGE_SIZE / max(h, w)
    if scale_factor < 1.0: img_resized = cv2.resize(enhanced_img, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_AREA)
    else: img_resized, scale_factor = enhanced_img, 1.0
    rh, rw = img_resized.shape[:2]
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    sam_start = time.time()
    masks = mask_generator.generate(img_rgb)
    print(f"SAM 原始生成了 {len(masks)} 个掩码 (耗时: {time.time() - sam_start:.2f}s)")
    seg_masks = [(m['segmentation'].astype('uint8') * 255) for m in masks]
    processed_masks = [psm for sm, m in zip(seg_masks, masks) for psm in (split_mask_watershed(sm) if m['area'] > SPLIT_AREA_THRESHOLD else [sm])]
    all_insts = compute_geoms_from_masks(processed_masks, scale_factor, rh, rw)
    if not all_insts: return None, None, 0
    insts_labeled, _ = cluster_and_label(all_insts)
    sorted_known = sorted(KNOWN_DIAMETERS_MM)
    counts = {str(mm): 0 for mm in sorted_known}
    for it in insts_labeled:
        if 'label' in it and it['label'] < len(sorted_known):
            counts[str(sorted_known[it['label']])] += 1
            
    # ==================== 核心修改点 2 (开始) ====================
    # 调用 visualize 函数时，传入 counts 和总数 len(all_insts)
    output_image = visualize(img, insts_labeled, sorted_known, counts, len(all_insts))
    # ==================== 核心修改点 2 (结束) ====================
    
    return counts, output_image, len(all_insts)


async def handler(websocket, path):
    """
    WebSocket 连接的处理函数。
    - ... (说明不变) ...
    - 将处理结果（统计数据）打包成 JSON 格式。
    - 将可视化图像保存在服务器本地。
    - ...
    """
    print(f"客户端 {websocket.remote_address} 已连接 (路径: {path})."); global mask_generator
    # <--- 新增：确保服务器输出目录存在
    if not os.path.exists(SERVER_OUTPUT_DIR):
        os.makedirs(SERVER_OUTPUT_DIR)
        print(f"已创建服务器输出目录: {SERVER_OUTPUT_DIR}")
        
    try:
        async for message in websocket:
            total_start_time = time.time()
            nparr = np.frombuffer(message, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None: print("接收到无效的图像数据"); continue
            
            print(f"\n收到来自 {websocket.remote_address} 的一帧图像，尺寸: {img.shape}")

            # ==================== 核心修改点 ====================
            # 在一个独立的线程中运行阻塞的 process_frame 函数
            stats, result_img, total_count = await asyncio.to_thread(
                process_frame, img, mask_generator
            )
            # ======================================================

               # 如果处理成功，准备成功响应
            if stats is not None and result_img is not None:
                # ==================== 核心修改点 (开始) ====================
                # 1. 不再编码图片为 Base64
                # _, buffer = cv2.imencode('.jpg', result_img)         # <--- 删除
                # img_base64 = base64.b64encode(buffer).decode('utf-8') # <--- 删除

                # 2. 将结果图片保存在服务器本地
                filename = f"result_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                output_path = os.path.join(SERVER_OUTPUT_DIR, filename)
                cv2.imwrite(output_path, result_img)
                print(f"✅ 分割效果图已保存至服务器: {output_path}")

                # 3. 构造不含图片的 JSON 响应
                response = {"status": "success", "total_count": total_count, "stats": stats} # <--- 修改
                # ==================== 核心修改点 (结束) ====================
            else:
                response = {"status": "error", "message": "在图像中未检测到合格的球体。"}
            
            processing_time = time.time() - total_start_time
            response['processing_time'] = f"{processing_time:.2f}"
            
            # ... (日志记录部分不变) ...

            # 发送 JSON 响应
            await websocket.send(json.dumps(response))
            print(f"结果已发回，总处理耗时: {processing_time:.2f} 秒")

    except websockets.exceptions.ConnectionClosed as e: print(f"客户端 {websocket.remote_address} 连接已断开: {e}")
    except Exception as e: print(f"处理时发生未预料的错误: {e}")
    finally: print(f"客户端 {websocket.remote_address} 会话结束.")

async def main():
    """
    服务器主函数。
    - 检查模型文件是否存在。
    - 加载 SAM 模型到指定设备（GPU 或 CPU）。
    - 创建 SamAutomaticMaskGenerator 实例。
    - 启动 WebSocket 服务器并永久运行，等待客户端连接。
    """
    print("=============================================="); print("             球体粒径估计推理服务器             "); print("==============================================")
    if not os.path.exists(SAM_CHECKPOINT): print(f"错误: SAM checkpoint 文件未找到: '{SAM_CHECKPOINT}'"); return
    print(f"Device: {DEVICE}"); print("正在加载 SAM 模型 (这可能需要一些时间)...")
    # 注册并加载模型
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE) # 将模型移动到 GPU 或 CPU
    global mask_generator
    mask_generator = SamAutomaticMaskGenerator(sam, **SAM_GEN_CFG)
    print("✅ SAM 模型加载完毕!")
    # 启动 WebSocket 服务，max_size 设置了允许接收的最大消息大小（这里是 2^24 字节 ≈ 16MB）
    async with websockets.serve(handler, HOST, PORT, max_size=2**24):
        print(f"✅ WebSocket 服务器已在 ws://{HOST}:{PORT} 启动"); print("等待 Atlas 客户端连接...")
        await asyncio.Future() # 永久运行，直到被中断

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # 捕获 Ctrl+C 中断信号，优雅地关闭服务器
        print("\n服务器正在关闭...")
    finally:
        # 在程序退出前，如果使用了 CUDA，清空 GPU 缓存
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()