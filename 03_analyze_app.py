import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
from analyze import analyze_bp, init_app as init_analyze
import socket
from werkzeug.utils import secure_filename
from image_processor import ImageProcessor
import cv2
import numpy as np
import base64

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__, static_url_path='', static_folder='static')
    
    app.config.update(
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
        UPLOAD_FOLDER='uploads',
        SECRET_KEY=os.urandom(24),
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
        PERMANENT_SESSION_LIFETIME=1800
    )
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    try:
        app.register_blueprint(analyze_bp)
        init_analyze(app)
        return app
    except Exception as e:
        logger.error(f"创建应用时出错: {str(e)}")
        raise

app = create_app()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'dwg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

image_processor = ImageProcessor()

@app.route('/')
def index():
    return render_template('drawFlow.html')

@app.route('/poufen')
def poufen():
    return render_template('poufen.html')

def compress_image(image: np.ndarray, max_size_mb: float = 0.8) -> np.ndarray:
    try:
        quality = 95
        while True:
            _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            size_mb = len(buffer) / 1024 / 1024
            
            if size_mb <= max_size_mb or quality <= 30:
                break
                
            quality -= 5
            
        logger.info(f"图片压缩完成，质量：{quality}，大小：{size_mb:.2f}MB")
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"图片压缩失败: {str(e)}")
        raise

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': '没有上传图片'
            }), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '未选择图片'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': '不支持的文件格式'
            }), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError("无法读取图片数据")
            
            processed_image = image_processor.enhance_image(image)
            processed_image = compress_image(processed_image, max_size_mb=0.8)
            logger.info("图像处理成功")
            
            _, buffer = cv2.imencode('.jpg', processed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            
            from analyze import analyze_uploaded_image
            result = analyze_uploaded_image(buffer.tobytes())
            
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"图像处理失败: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'success': False,
                'error': f'图像处理失败: {str(e)}'
            }), 500

    except Exception as e:
        logger.error(f"处理图片时发生错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'处理图片失败: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'success': False,
        'error': '请求的资源不存在'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': '服务器内部错误'
    }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'success': False,
        'error': '上传的文件太大'
    }), 413

def start_server():
    try:
        start_port = 5000
        end_port = 5051
        
        for port in range(start_port, end_port):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))
                    
                logger.info(f"服务器正在启动，端口: {port}")
                logger.info(f"请访问: http://127.0.0.1:{port}")
                logger.info("建筑图通·图纸流程检索系统已启动")
                
                app.run(host='127.0.0.1', port=port, debug=True)
                return True
                
            except OSError:
                logger.info(f"端口 {port} 已被占用，尝试下一个端口...")
                continue
                
        logger.error("无法找到可用端口")
        return False
        
    except Exception as e:
        logger.error(f"启动服务器时出错: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        if not start_server():
            logger.error("服务器启动失败")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n正在关闭服务器...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"发生未知错误: {str(e)}")
        sys.exit(1)