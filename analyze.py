# -*- coding: utf-8 -*-
import logging
from flask import Blueprint, render_template, request, jsonify, make_response, current_app
import cv2
import numpy as np
import base64
import requests
from image_processor import ImageProcessor
import json
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from graphviz import Digraph
import struct
import os
import time
import socket
import urllib3
from test import extract_general_notes
import re
import mistune

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

analyze_bp = Blueprint('dwg_analysis_blueprint', __name__,
                       template_folder='templates',
                       static_folder='static')

API_KEY = 'b2ce30c9-d036-4c5e-8b6f-74c3219ee576'
API_BASE_URL = 'https://ark.cn-beijing.volces.com'
API_PATH = '/api/v3/chat/completions'
API_MODEL = 'deepseek-v3-250324'
MAX_BASE64_LENGTH = 130000

retry_strategy = Retry(
    total=3,
    backoff_factor=5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(['GET', 'POST']),
    raise_on_status=False
)

class CustomHTTPAdapter(HTTPAdapter):
    def __init__(self, *args, **kwargs):
        self.socket_options = kwargs.pop('socket_options', None)
        super(CustomHTTPAdapter, self).__init__(*args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        if self.socket_options is None:
            self.socket_options = [
                (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
                (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
                (socket.SOL_SOCKET, socket.SO_REUSEADDR, 1),
            ]
        kwargs['socket_options'] = self.socket_options
        kwargs['retries'] = retry_strategy
        kwargs['timeout'] = 30
        super(CustomHTTPAdapter, self).init_poolmanager(*args, **kwargs)

def create_session():
    session = requests.Session()
    adapter = CustomHTTPAdapter(
        pool_connections=5,
        pool_maxsize=5,
        max_retries=retry_strategy,
        pool_block=True
    )

    session.timeout = (60, 300)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def call_api(prompt, max_retries=5):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "Connection": "keep-alive",
        "Keep-Alive": "timeout=60, max=1000"
    }

    payload = {
        "model": API_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2000
    }

    session = create_session()
    last_exception = None
    wait_time = 1

    for attempt in range(max_retries):
        try:
            response = session.post(
                f"{API_BASE_URL}{API_PATH}",
                headers=headers,
                json=payload,
                timeout=(30, 120)
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', wait_time))
                logger.warning(f"请求被限流，等待 {retry_after} 秒后重试")
                time.sleep(retry_after)
                wait_time = min(wait_time * 2, 60)
                continue
            else:
                error_msg = f"API调用失败: HTTP {response.status_code}"
                try:
                    error_detail = response.json()
                    if "error" in error_detail:
                        error_msg = f"{error_msg} - {error_detail['error'].get('message', '')}"
                except:
                    pass
                raise requests.exceptions.RequestException(error_msg)

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                urllib3.exceptions.ProtocolError,
                socket.error) as e:
            last_exception = e
            wait_time = min(wait_time * 2, 60)
            logger.warning(f"第 {attempt + 1} 次请求失败: {str(e)}, 等待 {wait_time} 秒后重试")
            time.sleep(wait_time)
            continue

        except Exception as e:
            logger.error(f"API调用出错: {str(e)}")
            raise
        finally:
            session.close()

    if last_exception:
        raise last_exception
    raise Exception("达到最大重试次数")

def compress_image(image, target_size):
    quality = 60
    max_size = 1024
    min_size = 256

    while True:
        height, width = image.shape[:2]
        if height > max_size or width > max_size:
            if height > width:
                new_height = max_size
                new_width = int(width * (max_size / height))
            else:
                new_width = max_size
                new_height = int(height * (max_size / width))
            image = cv2.resize(image, (new_width, new_height))

        _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        image_b64 = base64.b64encode(buffer).decode('utf-8')

        if len(image_b64) <= target_size or max_size <= min_size:
            break

        if len(image_b64) > target_size:
            if quality > 20:
                quality -= 10
            else:
                max_size = int(max_size * 0.8)
                quality = 60

    return image, image_b64

def preprocess_image(image):
    try:
        original_height, original_width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        alpha = 1.5
        beta = 10
        enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        equalized = clahe.apply(enhanced)
        denoised = cv2.GaussianBlur(equalized, (3, 3), 0)
        edges = cv2.Canny(denoised, 100, 200)
        kernel_dilate = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(edges, kernel_dilate, iterations=1)
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        combined = cv2.addWeighted(binary, 0.7, dilated, 0.3, 0)
        kernel_close = np.ones((2, 2), np.uint8)
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
        denoised_final = cv2.medianBlur(closed, 3)
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised_final, -1, kernel_sharpen)
        processed = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        if processed.shape[:2] != (original_height, original_width):
            processed = cv2.resize(processed, (original_width, original_height))
        final_image = cv2.convertScaleAbs(processed, alpha=1.2, beta=5)
        logger.info("图像预处理完成")
        return final_image

    except Exception as e:
        logger.error(f"图像预处理失败: {str(e)}")
        return image

def analyze_uploaded_image(image_bytes):
    try:
        logger.info("开始分析图片...")
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("无法读取图片数据，请确保上传的是有效的图片文件")

        height, width = image.shape[:2]
        logger.info(f"原始图片尺寸: {width}x{height}")

        processed_image = preprocess_image(image)
        logger.info("完成图像预处理")

        processed_image, image_b64 = compress_image(processed_image, MAX_BASE64_LENGTH)
        logger.info(f"压缩后的base64长度: {len(image_b64)}")

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        }

        extract_prompt = """请仔细分析这张工程图纸，提取所有可见的文字信息。\n\n提取要求：\n请完整、准确、专业地提取图纸中所有可见的文字信息，具体包括：\n\n一、标题栏信息：\n  1. 图纸名称及编号\n  2. 设计、审核、批准人员\n  3. 日期、比例等相关信息\n\n二、图纸内容：\n  1. 所有尺寸标注\n  2. 所有技术要求\n  3. 所有注释说明\n  4. 所有图例符号说明\n  5. 所有表格内容\n\n三、提取规则：\n  1. 按照从上到下、从左到右的顺序依次提取\n  2. 保持原始格式和布局，表格内容请以清晰的表格结构呈现\n  3. 标注具体位置（如上方、下方、左侧、右侧等）\n  4. 对于模糊或无法辨认的文字，请用\"模糊\"字样标注\n  5. 如果发现类似"12G614-1第1页"、"12G614-1第2页"等编号与页码紧密组合的信息，请完整、准确地提取，并在输出中明确标注编号及对应页码（页码请统一用'第X页'的中文格式，不要用01、02等数字格式）。\n\n输出要求：\n请以纯文本、整齐、专业、规范的格式直接列出所有提取到的文字内容，不要使用任何Markdown、星号、井号、短横线等符号，不要添加任何解释说明。内容应便于直接用于工程报告。"""

        payload = {
            "model": API_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个工程图纸文字识别助手。请直接列出图纸中看到的所有文字，保持原始格式和位置顺序，不要添加任何解释。"
                },
                {
                    "role": "user",
                    "content": f"{extract_prompt}\n\n图片数据：data:image/jpeg;base64,{image_b64}"
                }
            ],
            "temperature": 0.1,
            "max_tokens": 4000,
            "top_p": 0.99
        }

        logger.info(f"正在发送API请求到 {API_BASE_URL}")

        try:
            response = requests.post(
                API_BASE_URL,
                headers=headers,
                json=payload,
                timeout=(30, 180)
            )

            logger.info(f"API响应状态码: {response.status_code}")
            logger.info(f"API响应头: {response.headers}")

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    extracted_content = result['choices'][0]['message']['content'].strip()
                    logger.info("成功获取API响应内容")

                    if not extracted_content:
                        raise ValueError("API返回的文字内容为空")

                    return {
                        'success': True,
                        'analysis': extracted_content
                    }
                else:
                    error_msg = 'API返回结果格式错误'
                    logger.error(f"{error_msg}，返回内容：{result}")
                    return {
                        'success': False,
                        'error': error_msg
                    }
            else:
                error_msg = f'API调用失败: HTTP {response.status_code}'
                logger.error(f"{error_msg}")
                try:
                    error_detail = response.json()
                    if "error" in error_detail:
                        msg = error_detail.get("error", {}).get("message", "")
                        error_msg = f"{error_msg} - {msg}"
                    logger.error(f"API错误详情: {error_detail}")
                except:
                    logger.error(f"无法解析API错误响应: {response.text}")
                return {
                    'success': False,
                    'error': error_msg
                }

        except requests.exceptions.Timeout:
            error_msg = "API请求超时"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
        except requests.exceptions.RequestException as e:
            error_msg = f"API请求异常: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }

    except Exception as e:
        error_msg = f"分析图片时出错: {str(e)}"
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg
        }

def analyze_uploaded_images(image_files):
    results = []
    for file in image_files:
        if file.filename == '':
            continue

        try:
            image_bytes = file.read()
            result = analyze_uploaded_image(image_bytes)
            results.append({
                'filename': file.filename,
                'result': result
            })
        except Exception as e:
            logger.error(f"处理图片 {file.filename} 时出错: {str(e)}")
            results.append({
                'filename': file.filename,
                'result': {
                    'success': False,
                    'error': f'处理失败: {str(e)}'
                }
            })

    return results

def create_json_response(data, status_code=200):
    try:
        response = make_response(jsonify(data))
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        response.status_code = status_code
        return response
    except Exception as e:
        logger.error(f"创建JSON响应时出错: {str(e)}")
        error_response = make_response(jsonify({
            'success': False,
            'error': '服务器错误',
            'details': str(e)
        }))
        error_response.headers['Content-Type'] = 'application/json; charset=utf-8'
        error_response.status_code = 500
        return error_response

@analyze_bp.route('/poufen')
def poufen_page():
    return render_template('poufen.html')

@analyze_bp.route('/process_image', methods=['POST'])
def process_image():
    try:
        logger.info("接收到图片处理请求")

        if not request.content_type:
            logger.error("请求没有Content-Type")
            return create_json_response({
                'success': False,
                'error': '无效的请求格式',
                'details': '请求缺少Content-Type',
                'results': []
            }, 400)

        if 'multipart/form-data' not in request.content_type.lower():
            logger.error(f"无效的Content-Type: {request.content_type}")
            return create_json_response({
                'success': False,
                'error': '无效的请求格式，需要multipart/form-data格式',
                'details': f'当前Content-Type: {request.content_type}',
                'results': []
            }, 400)

        if 'image' not in request.files:
            logger.error("请求中没有图片文件")
            return create_json_response({
                'success': False,
                'error': '没有上传图片',
                'details': '请求中未找到image字段',
                'results': []
            }, 400)

        files = request.files.getlist('image')
        if not files or all(file.filename == '' for file in files):
            logger.error("没有选择文件")
            return create_json_response({
                'success': False,
                'error': '未选择文件',
                'details': '请选择要上传的图片文件',
                'results': []
            }, 400)

        results = []
        total_files = len(files)
        processed_files = 0

        for file in files:
            try:
                logger.info(f"开始处理文件: {file.filename}")

                if not file.content_type:
                    logger.warning(f"文件没有Content-Type: {file.filename}")
                    results.append({
                        'filename': file.filename,
                        'error': '无法确定文件类型',
                        'success': False
                    })
                    continue

                if 'image' not in file.content_type.lower():
                    logger.warning(f"文件类型无效: {file.filename} ({file.content_type})")
                    results.append({
                        'filename': file.filename,
                        'error': f"文件 {file.filename} 不是有效的图片格式",
                        'details': f'文件类型: {file.content_type}',
                        'success': False
                    })
                    continue

                try:
                    image_bytes = file.read()
                    if not image_bytes:
                        raise ValueError("文件为空")

                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError("无法解码图片数据")

                    _, buffer = cv2.imencode('.jpg', img)
                    image_b64 = base64.b64encode(buffer).decode('utf-8')

                    results.append({
                        'filename': file.filename,
                        'processed_image_url': f'data:image/jpeg;base64,{image_b64}',
                        'success': True
                    })

                    processed_files += 1
                    logger.info(f"成功处理图片 {processed_files}/{total_files}: {file.filename}")

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"处理图片数据时出错: {error_msg}")
                    results.append({
                        'filename': file.filename,
                        'error': f"处理图片数据失败: {error_msg}",
                        'success': False
                    })

            except Exception as e:
                error_msg = str(e)
                logger.error(f"处理图片 {file.filename} 时出错: {error_msg}")
                results.append({
                    'filename': file.filename,
                    'error': error_msg,
                    'success': False
                })

        success_count = sum(1 for result in results if result.get('success', False))

        response_data = {
            'success': success_count > 0,
            'message': f'成功处理 {success_count}/{total_files} 个文件',
            'results': results
        }

        if success_count == 0:
            response_data['error'] = '所有图片处理失败'
            return create_json_response(response_data, 400)

        logger.info(f"图片处理完成: {success_count}/{total_files} 个文件成功")
        return create_json_response(response_data)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"处理图片时出错: {error_msg}")
        return create_json_response({
            'success': False,
            'error': error_msg,
            'details': '服务器处理请求时发生错误',
            'results': []
        }, 500)

@analyze_bp.errorhandler(413)
def request_entity_too_large(error):
    return create_json_response({
        'success': False,
        'error': '上传的文件太大',
        'results': []
    }, 413)

@analyze_bp.errorhandler(500)
def internal_server_error(error):
    return create_json_response({
        'success': False,
        'error': '服务器内部错误',
        'results': []
    }, 500)

@analyze_bp.errorhandler(405)
def method_not_allowed(error):
    return create_json_response({
        'success': False,
        'error': '不支持的请求方法',
        'results': []
    }, 405)

@analyze_bp.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        logger.info("收到分析图片请求")

        if not request.content_type:
            logger.error("请求缺少Content-Type")
            return create_json_response({
                'success': False,
                'error': '无效的请求格式：缺少Content-Type',
                'analysis': ''
            }, 400)

        logger.info(f"请求Content-Type: {request.content_type}")

        if 'multipart/form-data' not in request.content_type:
            logger.error(f"无效的Content-Type: {request.content_type}")
            return create_json_response({
                'success': False,
                'error': '无效的请求格式：需要multipart/form-data格式',
                'analysis': ''
            }, 400)

        if 'image' not in request.files:
            logger.error("请求中没有图片文件")
            return create_json_response({
                'success': False,
                'error': '没有上传图片：请求中未找到image字段',
                'analysis': ''
            }, 400)

        files = request.files.getlist('image')
        logger.info(f"接收到 {len(files)} 个文件")

        if not files or all(file.filename == '' for file in files):
            logger.error("没有选择有效的文件")
            return create_json_response({
                'success': False,
                'error': '未选择文件：请选择要上传的图片',
                'analysis': ''
            }, 400)

        results = []
        total_files = len(files)
        processed_files = 0

        for file in files:
            try:
                logger.info(f"开始处理文件：{file.filename}")
                logger.info(f"文件类型：{file.content_type}")

                if not file.content_type:
                    raise ValueError(f"文件 {file.filename} 没有指定类型")

                if 'image' not in file.content_type:
                    raise ValueError(f"文件 {file.filename} 不是图片格式（{file.content_type}）")

                image_bytes = file.read()
                if not image_bytes:
                    raise ValueError(f"文件 {file.filename} 为空")

                logger.info(f"成功读取文件：{file.filename}，大小：{len(image_bytes)} 字节")

                result = analyze_uploaded_image(image_bytes)
                results.append({
                    'filename': file.filename,
                    'result': result
                })

                processed_files += 1
                logger.info(f"成功分析图片 {processed_files}/{total_files}: {file.filename}")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"分析图片 {file.filename} 时出错: {error_msg}")
                results.append({
                    'filename': file.filename,
                    'result': {
                        'success': False,
                        'error': error_msg
                    }
                })

        if not results:
            logger.error("没有可分析的图片")
            return create_json_response({
                'success': False,
                'error': '没有可分析的图片',
                'analysis': ''
            }, 400)

        extracted_text = ""
        success_count = 0

        for idx, result in enumerate(results, 1):
            if result['result']['success']:
                content = result['result']['analysis']
                extracted_text += content + "\n"
                success_count += 1
            else:
                extracted_text += f"提取失败：{result['result']['error']}\n"

        if success_count == 0:
            logger.error("所有图片分析失败")
            return create_json_response({
                'success': False,
                'error': '所有图片分析失败',
                'analysis': extracted_text
            }, 400)

        logger.info(f"分析完成：成功处理 {success_count}/{total_files} 个文件")
        return create_json_response({
            'success': True,
            'message': f'成功分析 {success_count}/{total_files} 个文件',
            'analysis': extracted_text
        })

    except Exception as e:
        error_msg = str(e)
        logger.error(f"分析图片时出错: {error_msg}")
        return create_json_response({
            'success': False,
            'error': error_msg,
            'analysis': ''
        }, 500)

def init_app(app):
    try:
        if 'dwg_analysis_blueprint' not in app.blueprints:
            app.register_blueprint(analyze_bp)
            logger.info("成功注册DWG分析蓝图")

        app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
        app.config['UPLOAD_FOLDER'] = 'uploads'
        app.config['PERMANENT_SESSION_LIFETIME'] = 1800

        @app.after_request
        def after_request(response):
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
            response.headers.add('Connection', 'keep-alive')
            response.headers.add('Keep-Alive', 'timeout=5, max=1000')
            return response

        @app.errorhandler(500)
        def internal_server_error(error):
            logger.error(f"服务器错误: {str(error)}")
            return jsonify({
                'success': False,
                'error': '服务器内部错误，请稍后重试'
            }), 500

        @app.errorhandler(ConnectionError)
        def handle_connection_error(error):
            logger.error(f"连接错误: {str(error)}")
            return jsonify({
                'success': False,
                'error': '连接服务器失败，请检查网络连接'
            }), 503

        return app

    except Exception as e:
        logger.error(f"初始化应用时出错: {str(e)}")
        raise

def generate_mindmap(report_text):
    try:
        mindmap_data = parse_analysis_to_mindmap(report_text)
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'mindmaps')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mindmap_json_path = os.path.join(output_dir, f'mindmap_{timestamp}.json')
        with open(mindmap_json_path, 'w', encoding='utf-8') as f:
            json.dump(mindmap_data, f, ensure_ascii=False, indent=2)
        return {
            'success': True,
            'mindmap_path': f'/static/mindmaps/mindmap_{timestamp}.json'
        }
    except Exception as e:
        logger.error(f"生成思维导图时出错: {str(e)}")
        return {
            'success': False,
            'error': f'生成思维导图失败: {str(e)}'
        }

def parse_analysis_to_mindmap(report_text):
    import re
    lines = report_text.splitlines()
    mindmap = {
        'name': '建筑工程图纸编号分析报告',
        'color': '#4682b4',
        'description': '',
        'children': []
    }

    def parse_nodes(start_idx, level):
        nodes = []
        i = start_idx
        while i < len(lines):
            line = lines[i]
            m = re.match(r'^(#+)\s+(.+)', line)
            if m:
                cur_level = len(m.group(1))
                title = m.group(2).strip()
                if cur_level < level:
                    break
                elif cur_level >= level:
                    node = {
                        'name': title,
                        'color': '#4CAF50',
                        'description': '',
                        'children': []
                    }
                    content = []
                    j = i + 1
                    while j < len(lines):
                        m2 = re.match(r'^(#+)\s+(.+)', lines[j])
                        if m2:
                            next_level = len(m2.group(1))
                            if next_level <= cur_level:
                                break
                            else:
                                break
                        content.append(lines[j])
                        j += 1
                    content_text = '\n'.join(content).strip()
                    if content_text:
                        node['children'].append({
                            'name': '内容',
                            'color': '#888',
                            'description': content_text,
                            'children': []
                        })
                    nodes.append(node)
                    i = j
                    continue
            i += 1
        return nodes, i

    first_idx = 0
    for idx, line in enumerate(lines):
        if re.match(r'^#+\s+', line):
            first_idx = idx
            break
    children, _ = parse_nodes(first_idx, 1)
    if children:
        mindmap['children'] = children
    else:
        mindmap['description'] = report_text.strip()
    return mindmap

@analyze_bp.route('/analyze_dwg', methods=['POST'])
def analyze_dwg():
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': '无效的请求格式：需要JSON数据'
            }), 400

        data = request.get_json()
        if not data or 'file_path' not in data:
            return jsonify({
                'success': False,
                'error': '请提供文件路径'
            }), 400

        file_path = data['file_path']
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': f'文件不存在: {file_path}'
            }), 404

        try:
            content = process_file_content(file_path)
            if not content or content.startswith("处理文件时出错") or content == "未找到设计总说明相关内容":
                return jsonify({
                    'success': False,
                    'error': content if content else "无法提取文件内容"
                }), 400

            try:
                number_info = extract_number_info(content)
                if not isinstance(number_info, dict):
                    logger.error(f"提取的编号信息不是字典类型: {type(number_info)}")
                    number_info = {
                        'standard': [],
                        'drawing': [],
                        'component': [],
                        'node': [],
                        'design': [],
                        'other': []
                    }
            except Exception as e:
                logger.error(f"提取编号信息时出错: {str(e)}")
                number_info = {
                    'standard': [],
                    'drawing': [],
                    'component': [],
                    'node': [],
                    'design': [],
                    'other': []
                }

            number_summary = "# 提取的编号信息\n\n"

            for category, items in number_info.items():
                if not isinstance(items, list):
                    logger.warning(f"编号信息中的{category}不是列表类型，跳过处理")
                    continue

                if items:
                    category_names = {
                        'standard': '标准规范编号',
                        'drawing': '图纸编号',
                        'component': '构件编号',
                        'node': '节点编号',
                        'design': '设计图集编号',
                        'other': '其他编号'
                    }
                    number_summary += f"## {category_names.get(str(category), str(category))}\n\n"
                    for item in items:
                        if isinstance(item, dict) and 'number' in item and 'context' in item:
                            number_summary += f"- **{item['number']}**: {item['context']}\n"
                    number_summary += "\n"

            content_chunks = split_content(content, max_chunk_size=50000)
            analysis_results = []

            has_numbers = False
            for category in number_info:
                if isinstance(number_info[category], list) and len(number_info[category]) > 0:
                    has_numbers = True
                    break

            if has_numbers:
                number_analysis_prompt = f"""请对以下建筑工程图纸中提取到的编号信息进行详细分析：

{number_summary}

请针对每个编号进行详细解释：
1. 编号的具体含义和所属标准
2. 编号在本工程中的具体应用
3. 对应的构件、材料或节点说明
4. 相关的技术参数和要求

请以markdown格式输出分析结果，并尽可能详细地解释每个编号的重要性和在图纸中的作用。"""

                try:
                    result = call_api(number_analysis_prompt)
                    if result and 'choices' in result and len(result['choices']) > 0:
                        number_analysis = result['choices'][0]['message']['content']
                        analysis_results.append(number_analysis)
                        logger.info(f"成功分析编号信息")
                    else:
                        logger.warning(f"编号信息分析返回空结果")
                except Exception as e:
                    logger.error(f"处理编号信息时出错: {str(e)}")

            for i, chunk in enumerate(content_chunks):
                logger.info(f"正在处理第 {i + 1}/{len(content_chunks)} 块内容")

                prompt = f"""请分析以下建筑工程设计说明的部分内容，重点关注编号信息的提取和分析：

{chunk}

请以markdown格式输出分析结果，重点关注：
1. 标准编号信息（如：12G614-1、03G101-1等）
2. 图纸编号（如：建施-01、结施-02等）
3. 构件编号（如：KL-1、L-2等）
4. 节点编号（如：J-1、D-1等）
5. 其他重要编号信息

对于每个编号信息，请说明：
- 编号的含义和用途
- 适用的范围或条件
- 相关的技术要求或规范

请确保完整提取所有编号信息，并按照上述格式进行详细分析。"""

                try:
                    result = call_api(prompt)
                    if result and 'choices' in result and len(result['choices']) > 0:
                        analysis = result['choices'][0]['message']['content']
                        analysis_results.append(analysis)
                        logger.info(f"成功分析第 {i + 1} 块内容")
                    else:
                        logger.warning(f"第 {i + 1} 块内容分析返回空结果")
                except Exception as e:
                    logger.error(f"处理第 {i + 1} 块内容时出错: {str(e)}")
                    continue

                time.sleep(2)

            if not analysis_results:
                return jsonify({
                    'success': False,
                    'error': '未能成功分析任何内容'
                }), 500

            final_analysis = "\n\n---\n\n".join(analysis_results)

            def clean_report(text):
                lines = text.splitlines()
                
                first_content_index = -1
                for i, line in enumerate(lines):
                    stripped_line = line.strip()
                    if stripped_line.startswith('## ') or stripped_line.startswith('### '):
                        first_content_index = i
                        break
                
                if first_content_index != -1:
                    content_lines = lines[first_content_index:]
                else:
                    content_lines = [l for l in lines if not re.match(r'^#\s*.*(报告|分析)', l.strip())]

                final_lines = [line for line in content_lines if line.strip() != '```']
                cleaned_text = "\n".join(final_lines)
                cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()
                
                return "# 建筑工程图纸编号分析报告\n\n" + cleaned_text

            final_analysis = clean_report(final_analysis)

            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
            os.makedirs(output_dir, exist_ok=True)

            analysis_file = os.path.join(output_dir, "analysis_result.md")
            with open(analysis_file, "w", encoding="utf-8") as f:
                f.write(final_analysis)

            try:
                number_info_safe = {}
                for category, items in number_info.items():
                    category_str = str(category)
                    if isinstance(items, list):
                        items_safe = []
                        for item in items:
                            if isinstance(item, dict):
                                item_safe = {}
                                for k, v in item.items():
                                    item_safe[str(k)] = str(v)
                                items_safe.append(item_safe)
                        number_info_safe[category_str] = items_safe
                    else:
                        number_info_safe[category_str] = []

                number_info_file = os.path.join(output_dir, "number_info.json")
                with open(number_info_file, "w", encoding="utf-8") as f:
                    json.dump(number_info_safe, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"保存编号信息时出错: {str(e)}")

            return jsonify({
                'success': True,
                'analysis': final_analysis,
                'raw_content': content
            })

        except Exception as e:
            logger.error(f"分析过程中出错: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'分析失败: {str(e)}'
            }), 500

    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'服务器错误: {str(e)}'
        }), 500

@analyze_bp.route('/api/generate-mindmap', methods=['POST'])
def generate_mindmap_endpoint():
    try:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        analysis_file = os.path.join(output_dir, "analysis_result.md")

        if not os.path.exists(analysis_file):
            return jsonify({'success': False, 'error': '分析结果文件不存在，请先执行分析。'}), 404

        with open(analysis_file, "r", encoding="utf-8") as f:
            final_analysis = f.read()

        mindmap_result = generate_mindmap(final_analysis)
        if not mindmap_result.get("success"):
            return jsonify({'success': False, 'error': '生成思维导图失败。'}), 500

        mindmap_path = mindmap_result.get("mindmap_path")

        return jsonify({
            'success': True,
            'mindmap_path': mindmap_path
        })
    except Exception as e:
        logger.error(f"按需生成思维导图时出错: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'服务器错误: {str(e)}'
        }), 500

def split_content(content, max_chunk_size=50000):
    chunks = []
    current_chunk = []
    current_size = 0

    for line in content.split('\n'):
        line_size = len(line.encode('utf-8'))
        if current_size + line_size > max_chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_size = 0
        current_chunk.append(line)
        current_size += line_size

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks

def process_file_content(file_path):
    try:
        if not isinstance(file_path, str):
            file_path = str(file_path)

        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.dwg':
            try:
                content = extract_general_notes(file_path)
                if not content or not isinstance(content, str):
                    logger.error(f"从DWG文件提取内容失败: {file_path}")
                    return "从DWG文件提取内容失败，请检查文件格式"

                try:
                    number_info = extract_number_info(content)
                    content += "\n\n=== 编号信息分析 ===\n"

                    for category, numbers in number_info.items():
                        if not isinstance(numbers, list):
                            continue

                        if numbers:
                            category_names = {
                                'standard': '标准规范编号',
                                'drawing': '图纸编号',
                                'component': '构件编号',
                                'node': '节点编号',
                                'design': '设计图集编号',
                                'other': '其他编号'
                            }
                            content += f"\n{category_names.get(str(category), str(category).upper())}:\n"
                            for item in numbers:
                                if isinstance(item, dict) and 'number' in item and 'context' in item:
                                    content += f"- {item['number']}: {item['context']}\n"
                except Exception as e:
                    logger.error(f"提取编号信息时出错: {str(e)}")

                return content
            except Exception as e:
                err_msg = f"处理DWG文件时出错: {str(e)}"
                logger.error(err_msg)
                return err_msg
        elif file_ext == '.pdf':
            try:
                content = extract_pdf_content(file_path)
                if not content or not isinstance(content, str):
                    logger.error(f"从PDF文件提取内容失败: {file_path}")
                    return "从PDF文件提取内容失败，请检查文件格式"
                return content
            except Exception as e:
                err_msg = f"处理PDF文件时出错: {str(e)}"
                logger.error(err_msg)
                return err_msg
        else:
            error_msg = f"不支持的文件类型: {file_ext}"
            logger.error(error_msg)
            return error_msg

    except Exception as e:
        error_msg = f"处理文件内容时出错: {str(e)}"
        logger.error(error_msg)
        return error_msg

def extract_pdf_content(file_path):
    try:
        import fitz

        doc = fitz.open(file_path)
        content = []

        for page in doc:
            text = page.get_text()
            if text.strip():
                content.append(text)

        doc.close()
        return "\n".join(content)

    except Exception as e:
        logger.error(f"提取PDF内容时出错: {str(e)}")
        return f"处理PDF文件时出错: {str(e)}"

@analyze_bp.route('/mindmap')
def mindmap_page():
    return render_template('mindmap.html')

@analyze_bp.route('/api/mindmap-data')
def get_mindmap_data():
    try:
        file_path = request.args.get('path')
        if not file_path:
            return jsonify({'error': '未提供文件路径'}), 400

        if file_path.startswith('/static/'):
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path.lstrip('/'))

        if not os.path.exists(file_path):
            return jsonify({'error': '文件不存在'}), 404

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            return jsonify({'error': '未找到思维导图数据'}), 404

        return jsonify(data)

    except Exception as e:
        logger.error(f"获取思维导图数据失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@analyze_bp.route('/api/import-xmind', methods=['POST'])
def import_xmind():
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': '未上传文件'
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '未选择文件'
            }), 400

        if not file.filename.endswith('.xmind'):
            return jsonify({
                'success': False,
                'error': '请上传XMind格式的文件'
            }), 400

        upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(upload_dir, f'xmind_{timestamp}.xmind')
        file.save(file_path)

        mindmap_data = parse_xmind_file(file_path)

        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'mindmaps')
        os.makedirs(output_dir, exist_ok=True)

        mindmap_json_path = os.path.join(output_dir, f'mindmap_{timestamp}.json')

        with open(mindmap_json_path, 'w', encoding='utf-8') as f:
            json.dump(mindmap_data, f, ensure_ascii=False, indent=2)

        logger.info(f"成功导入XMind文件: {file_path}")

        return jsonify({
            'success': True,
            'mindmap_path': f'/static/mindmaps/mindmap_{timestamp}.json'
        })

    except Exception as e:
        logger.error(f"导入XMind文件时出错: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'导入XMind文件失败: {str(e)}'
        }), 500

def parse_xmind_file(file_path):
    try:
        import zipfile
        import xml.etree.ElementTree as ET

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            content_xml = zip_ref.read('content.xml')

        root = ET.fromstring(content_xml)
        ns = {'xmap': 'urn:xmind:xmap:xmlns:content:2.0'}

        sheet = root.find('.//xmap:sheet', ns)
        if sheet is None:
            raise ValueError("未找到sheet元素")

        root_topic = sheet.find('.//xmap:topic', ns)
        if root_topic is None:
            raise ValueError("未找到根主题")

        mindmap_data = {
            'name': get_topic_title(root_topic, ns),
            'color': '#4682b4',
            'description': '从XMind导入的思维导图',
            'children': []
        }

        mindmap_data['children'] = process_xmind_topics(root_topic, ns)

        return mindmap_data

    except Exception as e:
        logger.error(f"解析XMind文件时出错: {str(e)}")
        raise

def get_topic_title(topic, ns):
    title_elem = topic.find('.//xmap:title', ns)
    if title_elem is not None and title_elem.text:
        return title_elem.text
    return "未命名主题"

def process_xmind_topics(topic, ns):
    children = []
    children_elem = topic.find('.//xmap:children', ns)
    if children_elem is None:
        return children

    topics_elem = children_elem.find('.//xmap:topics', ns)
    if topics_elem is None:
        return children

    for child_topic in topics_elem.findall('.//xmap:topic', ns):
        child_data = {
            'name': get_topic_title(child_topic, ns),
            'color': '#4CAF50',
            'description': '',
            'children': process_xmind_topics(child_topic, ns)
        }
        children.append(child_data)

    return children

DEEPSEEK_API_KEY = 'sk-55baa585a02a4129bc96756590570353'
DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions'
DEEPSEEK_MODEL = 'deepseek-v3-250324'

def deepseek_summarize(text, max_tokens=300):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "user", "content": f"请用简洁、专业的语言总结以下内容为一段纯文本：\n{text}"}
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens
    }
    try:
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        return text

def clean_content(text):
    text = re.sub(r'内容[:：]?', '', text)
    text = text.replace('*', '')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def generate_mindmap_plain_text_no_root(report_text, use_deepseek=False):
    lines = report_text.splitlines()
    segments = []
    buffer = []
    for line in lines:
        if line.strip() == '':
            if buffer:
                segments.append('\n'.join(buffer).strip())
                buffer = []
        else:
            buffer.append(line)
    if buffer:
        segments.append('\n'.join(buffer).strip())
    mindmap_nodes = []
    for idx, seg in enumerate(segments):
        if seg:
            text = deepseek_summarize(seg) if use_deepseek else seg
            text = clean_content(text)
            mindmap_nodes.append({
                'name': f'分支{idx+1}',
                'description': text,
                'children': []
            })
    return mindmap_nodes

def parse_markdown_ast(ast, level=1):
    result = []
    i = 0
    while i < len(ast):
        node = ast[i]
        if node['type'] == 'heading':
            title = clean_title(node['children'][0]['text'])
            children = []
            description = ''
            j = i + 1
            while j < len(ast):
                if ast[j]['type'] == 'heading' and ast[j]['level'] <= node['level']:
                    break
                if ast[j]['type'] == 'heading' and ast[j]['level'] > node['level']:
                    sub_nodes, skip = parse_markdown_ast(ast[j:], level+1)
                    children.extend(sub_nodes)
                    j += skip
                elif ast[j]['type'] == 'paragraph':
                    description += ast[j]['children'][0]['text'] + ' '
                    j += 1
                else:
                    j += 1
            result.append({
                'name': title,
                'description': split_and_format_description(description),
                'children': children
            })
            i = j
        else:
            i += 1
    return result, i

def clean_title(title):
    title = re.sub(r'^[\d一二三四五六七八九十百千万]+[\.、．\s]*', '', title)
    title = re.sub(r'^[\*\#\-\>\`]+', '', title)
    title = re.sub(r'：|:', '', title)
    title = re.sub(r'\s+', ' ', title)
    return title.strip()

def split_and_format_description(desc):
    desc = re.sub(r'[\*\#\-\>\`]', '', desc)
    desc = re.sub(r'\s+', ' ', desc)
    keywords = [
        '含义', '用途', '适用范围', '类型', '内容范围', '编号规则', '相关要求', '标准依据',
        '构件类型', '节点类型', '详细图索引', '图纸数量', '内容说明', '位置', '详图要求',
        '技术要求', '标准', '编号', '说明', '数量', '年限', '抗震设防烈度', '等级'
    ]
    pattern = '|'.join([f'({k})[：:]' for k in keywords])
    parts = re.split(pattern, desc)
    result = []
    idx = 1
    i = 1
    while i < len(parts):
        if parts[i] and i+1 < len(parts):
            key = parts[i].strip()
            value = parts[i+1].strip()
            if key and value:
                result.append(f"（{idx}）{key}：{value}")
                idx += 1
        i += 2
    if not result:
        desc = re.sub(r'\s+', ' ', desc)
        return desc.strip()
    return '\n'.join(result)

def main():
    input_path = input("请输入分析报告文本文件路径（如 output/analysis_result.txt ）: ").strip()
    if not os.path.exists(input_path):
        print("文件不存在！")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print("正在解析并生成思维导图结构...")
    tree = parse_report(text)

    output_dir = os.path.join(os.path.dirname(input_path), '../static/mindmaps')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'mindmap_{timestamp}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)
    print(f"思维导图已保存到: {output_path}")

@analyze_bp.route('/analyze')
def analyze_page():
    return render_template('drawFlow.html')

if __name__ == '__main__':
    main()