#!/usr/bin/env python3
"""
Ultralytics YOLO Gradio 前端界面
功能：目标检测、模型选择、模型测试日志、摄像头实时检测
"""

import os
import sys
import time
import json
import glob
import threading
from datetime import datetime
from pathlib import Path

import gradio as gr
import cv2
import numpy as np
from PIL import Image
import requests
import re

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR / 'ultralytics' / 'lib' / 'python3.11' / 'site-packages'))

from ultralytics import YOLO

DEFAULT_MODELS_DIR = str(BASE_DIR)
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

current_model = None
current_model_path = None

test_logs = []

last_detection_image = None
last_detection_data = []

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:1.5b"

camera_running = False
camera_capture = None
camera_frame_lock = threading.Lock()
current_camera_frame = None

def get_available_models():
    """获取可用的模型列表"""
    models = []
    for ext in ["*.pt", "*.pth"]:
        pattern = os.path.join(DEFAULT_MODELS_DIR, "**", ext)
        models.extend(glob.glob(pattern, recursive=True))
    models = list(set(models))
    models.sort()
    return models if models else ["未找到模型文件"]

def load_model(model_path):
    """加载选定的模型"""
    global current_model, current_model_path
    
    if not model_path or model_path == "未找到模型文件":
        return "[ERROR] 请先选择有效的模型文件"
    
    if not os.path.exists(model_path):
        return f"[ERROR] 模型文件不存在: {model_path}"
    
    try:
        current_model = YOLO(model_path)
        current_model_path = model_path
        model_name = os.path.basename(model_path)
        return f"[OK] 模型加载成功: {model_name}\n路径: {model_path}"
    except Exception as e:
        return f"[ERROR] 模型加载失败: {str(e)}"

def get_model_info():
    """获取当前模型的信息"""
    if current_model is None:
        return "尚未加载模型"
    
    try:
        return f"""
**模型信息：**
- 模型路径: {current_model_path}
- 任务类型: {getattr(current_model, 'task', 'unknown')}
- 模型类型: {getattr(current_model, 'type', 'unknown')}
        """
    except:
        return f"当前模型: {os.path.basename(current_model_path) if current_model_path else 'Unknown'}"

def detect_image(image, conf_threshold=0.25, iou_threshold=0.45):
    """对图片进行目标检测"""
    if current_model is None:
        return None, "[ERROR] 请先加载模型！"
    
    if image is None:
        return None, "[ERROR] 请先上传图片！"
    
    try:
        start_time = time.time()
        results = current_model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        inference_time = time.time() - start_time
        
        result = results[0]
        annotated_image = result.plot()
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        detections = []
        global last_detection_data, last_detection_image
        last_detection_data = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                class_name = result.names[cls_id]
                detections.append(f"{class_name}: {conf:.2%}")
                xyxy = box.xyxy[0].tolist()
                last_detection_data.append({
                    "class": class_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(x, 2) for x in xyxy]
                })
        last_detection_image = annotated_image.copy() if annotated_image is not None else None
        
        stats = f"""
**检测结果：**
- 检测对象数量: {len(detections)}
- 推理时间: {inference_time:.3f}s
- 置信度阈值: {conf_threshold}
- IoU阈值: {iou_threshold}
        """
        
        details = "\n".join(detections) if detections else "未检测到目标"
        
        log_entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "image",
            "model": os.path.basename(current_model_path) if current_model_path else "unknown",
            "detections": len(detections),
            "inference_time": f"{inference_time:.3f}s"
        }
        test_logs.append(log_entry)
        save_log(log_entry)
        
        return annotated_image, details
        
    except Exception as e:
        return None, f"[ERROR] 检测失败: {str(e)}"

def call_ollama(user_question=""):
    """调用本地 Ollama 分析检测结果"""
    global last_detection_data
    if not last_detection_data:
        return "[ERROR] 尚未进行目标检测，请先检测图片。"
    
    detection_text = json.dumps(last_detection_data, ensure_ascii=False, indent=2)
    system_prompt = f"""你是一位计算机视觉分析助手。以下是YOLO目标检测的结果，包含每个检测对象的类别、置信度和边界框坐标（格式为 [x1, y1, x2, y2]，对应左上角和右下角像素坐标）：

{detection_text}

请基于以上数据回答用户问题。分析时请考虑物体的位置关系、数量、置信度等信息。"""
    
    prompt = user_question if user_question else "请描述检测到的物体及其位置关系，并给出简要分析。"
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        answer = response.json().get("response", "AI 未返回有效回答")
        # 清理 DeepSeek-R1 的 <think> 思考过程标签
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        return answer
    except Exception as e:
        return f"[ERROR] AI 调用失败: {str(e)}"

def ai_quick_analyze():
    """一键分析当前检测结果"""
    return call_ollama()

def chat_with_ai(message, history):
    """AI 对话回调"""
    if not message.strip():
        return history, ""
    answer = call_ollama(message)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history, ""

def load_latest_detection():
    """加载最新检测结果到 AI 页面"""
    if last_detection_image is None:
        return None, "暂无检测数据，请先在「目标检测」页面进行检测。"
    return last_detection_image, json.dumps(last_detection_data, ensure_ascii=False, indent=2)

def detect_video(video, conf_threshold=0.25, iou_threshold=0.45):
    """对视频进行目标检测"""
    if current_model is None:
        return None, "[ERROR] 请先加载模型！"
    
    if video is None:
        return None, "[ERROR] 请先上传视频！"
    
    return None, "视频检测功能开发中...\n当前仅支持图片检测"

def camera_worker(device_id, conf_threshold, iou_threshold):
    """摄像头后台工作线程"""
    global camera_running, camera_capture, current_camera_frame
    
    camera_capture = cv2.VideoCapture(int(device_id))
    if not camera_capture.isOpened():
        return
    
    camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    last_time = time.time()
    fps = 0
    
    while camera_running:
        ret, frame = camera_capture.read()
        if not ret:
            continue
        
        frame_count += 1
        if frame_count % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - last_time) if current_time > last_time else 0
            last_time = current_time
        
        # 检测处理
        if current_model is not None:
            try:
                results = current_model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )
                result = results[0]
                frame = result.plot()
                det_count = len(result.boxes) if result.boxes is not None else 0
                cv2.putText(frame, f"Detections: {det_count}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                cv2.putText(frame, f"Error: {str(e)[:20]}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 显示FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        with camera_frame_lock:
            current_camera_frame = frame.copy()
    
    # 清理
    if camera_capture:
        camera_capture.release()
        camera_capture = None
    with camera_frame_lock:
        current_camera_frame = None

def start_camera(device_id, conf_threshold, iou_threshold):
    """启动摄像头"""
    global camera_running
    
    if current_model is None:
        return "[ERROR] 请先加载模型！", None
    
    if camera_running:
        return "摄像头已经在运行中", None
    
    camera_running = True
    thread = threading.Thread(
        target=camera_worker,
        args=(device_id, conf_threshold, iou_threshold)
    )
    thread.daemon = True
    thread.start()
    
    return "[OK] 摄像头已启动", None

def stop_camera():
    """停止摄像头"""
    global camera_running
    camera_running = False
    time.sleep(0.3)
    return "摄像头已停止"

def get_camera_frame():
    """获取当前摄像头帧 - 用于定时更新"""
    global current_camera_frame
    
    with camera_frame_lock:
        if current_camera_frame is not None:
            frame_rgb = cv2.cvtColor(current_camera_frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
    
    return None

def capture_camera_frame():
    """截取当前摄像头画面"""
    with camera_frame_lock:
        if current_camera_frame is not None:
            frame_rgb = cv2.cvtColor(current_camera_frame, cv2.COLOR_BGR2RGB)
            return frame_rgb, "[OK] 截图成功"
    return None, "[ERROR] 摄像头未运行"

def save_log(log_entry):
    """保存日志到文件"""
    log_file = LOGS_DIR / f"test_log_{datetime.now().strftime('%Y%m%d')}.json"
    
    logs = []
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            try:
                logs = json.load(f)
            except:
                logs = []
    
    logs.append(log_entry)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def get_logs():
    """获取所有日志"""
    if not test_logs:
        return "暂无测试日志"
    
    logs_text = []
    for log in reversed(test_logs[-50:]):
        logs_text.append(
            f"[{log['time']}] {log['type'].upper()} | "
            f"模型: {log['model']} | "
            f"检测数: {log['detections']} | "
            f"耗时: {log['inference_time']}"
        )
    
    return "\n".join(logs_text)

def clear_logs():
    """清空日志"""
    global test_logs
    test_logs = []
    return "日志已清空"

def export_logs():
    """导出日志到文件"""
    if not test_logs:
        return None, "暂无日志可导出"
    
    export_file = LOGS_DIR / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(export_file, 'w', encoding='utf-8') as f:
        json.dump(test_logs, f, ensure_ascii=False, indent=2)
    
    return str(export_file), f"[OK] 日志已导出到: {export_file}"

def create_ui():
    """创建 Gradio 界面"""
    
    with gr.Blocks(title="Ultralytics YOLO 检测平台") as demo:
        
        # ===== 标题 =====
        gr.Markdown("""
        # Ultralytics YOLO 目标检测平台
        
        基于 Gradio 的 YOLO 目标检测可视化界面 - 支持图片、视频和摄像头实时检测
        """)
        
        # ===== 标签页 =====
        with gr.Tabs():
            
            # ---------- 标签1: 目标检测 + AI 分析 ----------
            with gr.TabItem("目标检测 + AI 分析"):
                gr.Markdown("### 上传图片进行目标检测，并与 AI 对话分析结果")
                
                # ===== 左右两列布局 =====
                with gr.Row():
                    # --- 左列：图片检测（输入输出共用）---
                    with gr.Column(scale=1):
                        # 图片上传/结果显示（同一个组件）
                        input_image = gr.Image(
                            label="上传图片 / 检测结果",
                            type="numpy",
                            height=450
                        )
                        
                        with gr.Row():
                            conf_slider = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.25, step=0.05,
                                label="置信度阈值"
                            )
                            iou_slider = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.45, step=0.05,
                                label="IoU 阈值"
                            )
                        
                        detect_btn = gr.Button("开始检测", variant="primary")
                        
                        output_details = gr.Textbox(
                            label="检测详情",
                            lines=6,
                            interactive=False
                        )
                        
                        gr.Markdown("---")
                        gr.Markdown("#### AI 分析")
                        ai_quick_btn = gr.Button("一键分析", variant="primary")
                        ai_quick_output = gr.Markdown()
                    
                    # --- 右列：AI 对话 + 数据 ---
                    with gr.Column(scale=1):
                        gr.Markdown("#### AI 对话")
                        chatbot = gr.Chatbot(
                            label="DeepSeek-R1 1.5B",
                            height=400
                        )
                        with gr.Row():
                            chat_input = gr.Textbox(
                                label="输入问题",
                                placeholder="例如：这些物体之间有什么位置关系？",
                                scale=4
                            )
                            chat_send = gr.Button("发送", variant="primary", scale=1)
                        chat_clear = gr.Button("清空对话", variant="secondary")
                        
                        gr.Markdown("---")
                        gr.Markdown("#### 检测数据 (JSON)")
                        ai_detection_json = gr.Textbox(
                            label="",
                            lines=10,
                            interactive=False
                        )
                
                # 检测按钮点击后更新所有相关输出
                detect_btn.click(
                    fn=detect_image,
                    inputs=[input_image, conf_slider, iou_slider],
                    outputs=[input_image, output_details]
                )
                
                # AI 功能按钮绑定
                ai_quick_btn.click(fn=ai_quick_analyze, outputs=ai_quick_output)
                
                chat_send.click(
                    fn=chat_with_ai,
                    inputs=[chat_input, chatbot],
                    outputs=[chatbot, chat_input]
                )
                
                chat_clear.click(fn=lambda: ([], ""), outputs=[chatbot, chat_input])
                
                # 检测完成后自动更新 JSON 数据
                detect_btn.click(
                    fn=lambda: json.dumps(last_detection_data, ensure_ascii=False, indent=2) if last_detection_data else "暂无检测数据",
                    outputs=ai_detection_json
                )
            
            # ---------- 标签2: 摄像头实时检测 ----------
            with gr.TabItem("摄像头实时检测"):
                gr.Markdown("### 使用摄像头进行实时目标检测")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 控制面板")
                        
                        device_id = gr.Number(
                            value=0, label="摄像头设备ID",
                            info="通常为 0 (内置摄像头) 或 1, 2 (外接摄像头)",
                            precision=0
                        )
                        
                        camera_conf = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.25, step=0.05,
                            label="置信度阈值"
                        )
                        
                        camera_iou = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.45, step=0.05,
                            label="IoU 阈值"
                        )
                        
                        with gr.Row():
                            start_cam_btn = gr.Button("启动摄像头", variant="primary")
                            stop_cam_btn = gr.Button("停止摄像头", variant="stop")
                        
                        capture_btn = gr.Button("截取当前画面", variant="secondary")
                        
                        camera_status = gr.Textbox(
                            label="状态", value="摄像头未启动", interactive=False
                        )
                        
                        captured_image = gr.Image(
                            label="截取的图片", type="numpy", height=200
                        )
                        
                        gr.Markdown("""
                        #### 使用说明
                        1. 确保摄像头已连接
                        2. 设置好检测参数
                        3. 点击「启动摄像头」开始实时检测
                        4. 点击「截取当前画面」保存当前帧
                        5. 点击「停止摄像头」结束检测
                        
                        > [WARN] **注意**: 需要先加载模型才能进行检测
                        """)
                    
                    with gr.Column(scale=2):
                        camera_output = gr.Image(
                            label="实时检测画面",
                            height=480
                        )
                
                # 启动摄像头
                start_cam_btn.click(
                    fn=start_camera,
                    inputs=[device_id, camera_conf, camera_iou],
                    outputs=camera_status
                )
                
                # 停止摄像头
                stop_cam_btn.click(
                    fn=stop_camera,
                    outputs=camera_status
                )
                
                # 使用 Timer 定时更新画面 (Gradio 6.0+ 方式)
                timer = gr.Timer(value=0.05, active=False)
                timer.tick(fn=get_camera_frame, outputs=camera_output)
                
                # 启动/停止时控制 timer
                start_cam_btn.click(fn=lambda: gr.Timer(active=True), outputs=timer)
                stop_cam_btn.click(fn=lambda: gr.Timer(active=False), outputs=timer)
                
                # 截取画面
                capture_btn.click(
                    fn=capture_camera_frame,
                    outputs=[captured_image, camera_status]
                )
            
            # ---------- 标签3: 模型选择 ----------
            with gr.TabItem("模型选择"):
                gr.Markdown("### 选择并加载 YOLO 模型")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        model_dropdown = gr.Dropdown(
                            choices=get_available_models(),
                            label="选择模型",
                            info="从列表中选择要加载的模型文件"
                        )
                        
                        refresh_btn = gr.Button("刷新模型列表")
                        
                        model_path_input = gr.Textbox(
                            label="或输入模型路径",
                            placeholder="/path/to/your/model.pt"
                        )
                        
                        load_btn = gr.Button("加载模型", variant="primary")
                    
                    with gr.Column(scale=3):
                        load_status = gr.Textbox(
                            label="加载状态", lines=3, interactive=False
                        )
                        model_info = gr.Markdown(label="模型信息")
                
                refresh_btn.click(
                    fn=lambda: gr.Dropdown(choices=get_available_models()),
                    outputs=model_dropdown
                )
                
                def on_load_model(dropdown_val, text_val):
                    model_path = text_val if text_val else dropdown_val
                    return load_model(model_path)
                
                load_btn.click(
                    fn=on_load_model,
                    inputs=[model_dropdown, model_path_input],
                    outputs=load_status
                ).then(
                    fn=get_model_info,
                    outputs=model_info
                )
                
                default_models = get_available_models()
                if default_models and default_models[0] != "未找到模型文件":
                    gr.Markdown(f"""
                    > [TIP] **提示**: 项目目录下找到 {len(default_models)} 个模型文件
                    > 默认模型: `{os.path.basename(default_models[0])}`
                    """)
            
            # ---------- 标签4: 模型测试日志 ----------
            with gr.TabItem("模型测试日志"):
                gr.Markdown("### 查看测试历史和日志")
                
                with gr.Row():
                    refresh_log_btn = gr.Button("刷新日志")
                    clear_log_btn = gr.Button("清空日志", variant="stop")
                    export_log_btn = gr.Button("导出日志", variant="primary")
                
                log_display = gr.Textbox(
                    label="测试日志",
                    lines=20,
                    interactive=False,
                    value=get_logs()
                )
                
                export_file = gr.File(label="导出的日志文件")
                export_status = gr.Textbox(label="导出状态", interactive=False)
                
                refresh_log_btn.click(fn=get_logs, outputs=log_display)
                clear_log_btn.click(fn=clear_logs, outputs=log_display)
                
                def do_export():
                    path, msg = export_logs()
                    return path, msg
                
                export_log_btn.click(
                    fn=do_export,
                    outputs=[export_file, export_status]
                )
            
            # （原 AI 对话分析功能已合并到目标检测标签页）
        
        # ===== 底部信息 =====
        gr.Markdown("---")
        gr.Markdown("""
        <center>
        <b>Ultralytics YOLO Detection Platform</b> | 
        支持功能：图片检测 + AI 分析 | 视频检测 | 摄像头实时检测 | 模型管理 | 日志记录<br>
        Built with <a href="https://gradio.app">Gradio</a> + Ollama (DeepSeek-R1)
        </center>
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        theme=gr.themes.Soft()
    )
