from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from search import search_bp, init_search
from analyze import analyze_bp, init_app as init_analyze

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

app = Flask(__name__, 
            static_url_path='/static', 
            static_folder='static',
            template_folder='templates')

app.config.update(
    MAX_CONTENT_LENGTH=50 * 1024 * 1024,
    UPLOAD_FOLDER='uploads',
    SECRET_KEY=os.urandom(24).hex()
)

os.makedirs('uploads', exist_ok=True)
os.makedirs('pdf_files', exist_ok=True)
os.makedirs('output_images', exist_ok=True)
os.makedirs('temp_images', exist_ok=True)

model = YOLO('runs/detect/train_joint/weights/best.pt')
logger.info("YOLO模型加载成功")

app.register_blueprint(search_bp)
logger.info("成功注册图纸流程检索蓝图")

app.register_blueprint(analyze_bp)
logger.info("成功注册图纸智能分析蓝图")

init_analyze(app)
init_search()
logger.info("所有模块初始化完成")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognition')
def recognition():
    return render_template('recognition.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    try:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        results = model(img)
        r = results[0]
        
        annotated_img = r.plot()
        _, img_encoded = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        items = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            items.append({
                'name': cls_name,
                'confidence': f'{conf:.2f}'
            })
        
        return jsonify({
            'success': True,
            'annotated_image': img_base64,
            'detections': items,
            'total': len(items)
        })
    except Exception as e:
        logger.error(f"构件识别失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'识别失败: {str(e)}'
        }), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    from flask import send_from_directory
    return send_from_directory('static', filename)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"服务器内部错误: {error}")
    return jsonify({
        'success': False,
        'error': '服务器内部错误，请稍后重试'
    }), 500

if __name__ == '__main__':
    try:
        port = 5000
        logger.info("=" * 60)
        logger.info("建筑图通 · 智能标注平台 - 整合版")
        logger.info("=" * 60)
        logger.info(f"服务启动成功，端口: {port}")
        logger.info(f"访问地址: http://127.0.0.1:{port}")
        logger.info("-" * 60)
        logger.info("功能模块列表：")
        logger.info(f"1. 首页 - http://127.0.0.1:{port}")
        logger.info(f"2. 构件智能识别 - http://127.0.0.1:{port}/recognition")
        logger.info(f"3. 图纸流程检索 - http://127.0.0.1:{port}/search")
        logger.info(f"4. 图纸智能分析 - http://127.0.0.1:{port}/analyze")
        logger.info("-" * 60)
        logger.info("服务已就绪，按 Ctrl+C 停止服务")
        logger.info("=" * 60)
        
        app.run(host='127.0.0.1', port=port, debug=True, threaded=True)
            
    except KeyboardInterrupt:
        logger.info("\n正在关闭服务器...")
        logger.info("服务器已安全停止")
    except Exception as e:
        logger.error(f"启动服务器时出错: {str(e)}")
        sys.exit(1)