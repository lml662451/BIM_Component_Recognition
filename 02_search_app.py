import logging
from flask import Flask, render_template
from search import search_bp, init_search
import socket
import sys
import os

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
        SECRET_KEY=os.urandom(24),
    )

    try:
        app.register_blueprint(search_bp)

        @app.route('/')
        def index():
            return render_template('search_form.html')

        return app

    except Exception as e:
        logger.error(f"创建应用时出错: {str(e)}")
        raise

app = create_app()

@app.errorhandler(404)
def not_found_error(error):
    return render_template('search_form.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('search_form.html'), 500

def start_server():
    try:
        if not init_search():
            logger.error("搜索功能初始化失败")
            return False

        start_port = 5000
        end_port = 5051

        for port in range(start_port, end_port):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))

                logger.info(f"搜索服务启动成功，端口: {port}")
                logger.info(f"访问地址: http://127.0.0.1:{port}")

                app.run(host='127.0.0.1', port=port, debug=False)
                return True

            except OSError:
                logger.info(f"端口 {port} 已被占用，尝试下一个端口...")
                continue

        logger.error("无法找到可用端口（5000-5051）")
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
        logger.info("\n服务器已手动关闭")
        sys.exit(0)
    except Exception as e:
        logger.error(f"未知错误: {str(e)}")
        sys.exit(1)