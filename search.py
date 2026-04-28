import requests
import logging
from flask import Blueprint, render_template, request, jsonify, send_from_directory, send_file
import pymysql
import fitz
import os
import threading
import re

search_bp = Blueprint('search', __name__)

UPLOAD_FOLDER = 'pdf_files'
OUTPUT_FOLDER = 'output_images'
TEMP_FOLDER = 'temp_images'
ALLOWED_EXTENSIONS = {'pdf'}
KNOWLEDGE_BASE_PATH = r'12G614-1.pdf'

DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123',
    'database': 'pdf_storage',
    'charset': 'utf8mb4'
}

SIFLOW_API_KEY = 'sk-goeoulcsmgdllzbhwlhymihlbshcgxxkqcpfxgobtxewrduq'
SIFLOW_API_BASE_URL = 'https://api.siliconflow.cn/v1/chat/completions'

def init_database():
    try:
        with pymysql.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:
                cursor.execute("CREATE DATABASE IF NOT EXISTS pdf_storage")
                cursor.execute("USE pdf_storage")
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pdf_info (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        file_name VARCHAR(255) NOT NULL,
                        file_path VARCHAR(255) NOT NULL,
                        page_num INT,
                        image_path VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY unique_file (file_name, page_num)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pdf_images (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        pdf_id INT,
                        image_data LONGBLOB,
                        FOREIGN KEY (pdf_id) REFERENCES pdf_info(id) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)
                
                conn.commit()
                logging.info("数据库初始化成功")
                return True
    except Exception as e:
        logging.error(f"数据库初始化错误: {e}")
        return False

def process_pdf(pdf_path):
    try:
        filename = os.path.basename(pdf_path)
        logging.info(f"正在处理PDF文件: {filename}")
        
        base_name = os.path.splitext(filename)[0]
        output_folder = os.path.join(OUTPUT_FOLDER, base_name)
        os.makedirs(output_folder, exist_ok=True)
        logging.info(f"创建输出目录: {output_folder}")
        
        pdf_document = fitz.open(pdf_path)
        total_pages = pdf_document.page_count
        logging.info(f"PDF文件共有 {total_pages} 页")
        
        with pymysql.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM pdf_info WHERE file_name = %s", (filename,))
                
                for page_num in range(total_pages):
                    page = pdf_document.load_page(page_num)
                    zoom = 2.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    
                    image_filename = f"{base_name}_page_{page_num+1}.png"
                    image_full_path = os.path.join(output_folder, image_filename)
                    
                    pix.save(image_full_path)
                    logging.info(f"已保存图片: {image_full_path}")
                    
                    if os.path.exists(image_full_path):
                        logging.info(f"图片文件确认存在: {image_full_path}")
                        
                        cursor.execute("""
                            INSERT INTO pdf_info (file_name, file_path, page_num, image_path) 
                            VALUES (%s, %s, %s, %s)
                        """, (filename, pdf_path, page_num+1, image_filename))
                    else:
                        logging.error(f"图片保存失败: {image_full_path}")
                
                conn.commit()
        
        pdf_document.close()
        logging.info(f"完成处理PDF文件: {filename}")
        
    except Exception as e:
        logging.error(f"处理PDF文件时出错 {filename}: {e}")

def load_knowledge_base():
    try:
        with fitz.open(KNOWLEDGE_BASE_PATH) as pdf_document:
            knowledge_text = ""
            for page in pdf_document:
                knowledge_text += page.get_text()
        return knowledge_text
    except Exception as e:
        logging.error(f"加载知识库失败: {e}")
        return None

def ask_siflow(question):
    knowledge_text = load_knowledge_base()
    if not knowledge_text:
        return None
        
    try:
        headers = {
            "Authorization": f"Bearer {SIFLOW_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "Qwen/QwQ-32B",
            "messages": [
                {
                    "role": "system",
                    "content": "你需要根据提供的知识库信息回答用户问题"
                },
                {
                    "role": "user",
                    "content": f"知识库信息：{knowledge_text}。问题：{question}"
                }
            ],
            "stream": False,
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.7
        }
        
        response = requests.post(SIFLOW_API_BASE_URL, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            logging.error(f"API调用失败，状态码: {response.status_code}")
            return None
            
    except Exception as e:
        logging.error(f"调用硅基流动API时出错: {e}")
        return None

@search_bp.route('/search')
def search_page():
    return render_template('search_form.html')

@search_bp.route('/analyze_pdf', methods=['POST'])
def analyze_pdf():
    try:
        file_name = request.form.get('file_name')
        if not file_name:
            return jsonify({"error": "未提供文件名"}), 400
            
        if not file_name.endswith('.pdf'):
            file_name += '.pdf'
        
        base_name = os.path.splitext(file_name)[0]
        logging.info(f"查询PDF: {file_name}, 基础名称: {base_name}")
        
        pages = []
        source = "unknown"
        
        try:
            with pymysql.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT page_num, image_path 
                        FROM pdf_info 
                        WHERE file_name = %s AND page_num >= 5
                        ORDER BY page_num
                    """, (file_name,))
                    
                    results = cursor.fetchall()
                    
                    if results:
                        for row in results:
                            actual_page = row[0]
                            image_path = row[1]
                            display_page = actual_page - 4
                            
                            folder_path = os.path.join(OUTPUT_FOLDER, base_name)
                            image_full_path = os.path.join(folder_path, image_path)
                            
                            if os.path.exists(image_full_path):
                                pages.append({
                                    "page_number": display_page,
                                    "actual_page": actual_page,
                                    "image_path": image_path,
                                    "exists": True
                                })
                            else:
                                logging.warning(f"数据库记录但文件不存在: {image_full_path}")
                        
                        source = "database"
                        logging.info(f"从数据库找到 {len(pages)} 页")
        except Exception as e:
            logging.error(f"数据库查询失败: {e}")
        
        if len(pages) == 0:
            folder_path = os.path.join(OUTPUT_FOLDER, base_name)
            if os.path.exists(folder_path):
                try:
                    files = os.listdir(folder_path)
                    png_files = [f for f in files if f.endswith('.png')]
                    
                    page_files = []
                    for img_file in png_files:
                        match = re.search(r'page_(\d+)', img_file)
                        if match:
                            page_num = int(match.group(1))
                            if page_num >= 5:
                                page_files.append((page_num, img_file))
                    
                    page_files.sort(key=lambda x: x[0])
                    
                    for actual_page, img_file in page_files:
                        display_page = actual_page - 4
                        pages.append({
                            "page_number": display_page,
                            "actual_page": actual_page,
                            "image_path": img_file,
                            "exists": True
                        })
                    
                    if pages:
                        source = "filesystem"
                        logging.info(f"从文件系统找到 {len(pages)} 页")
                except Exception as e:
                    logging.error(f"读取文件夹失败: {e}")
        
        if pages:
            page_numbers = [p["page_number"] for p in pages]
            logging.info(f"返回页面: {page_numbers}")
            
            return jsonify({
                "success": True,
                "pages": pages,
                "source": source,
                "total": len(pages),
                "page_range": f"{pages[0]['page_number']}-{pages[-1]['page_number']}"
            })
        else:
            return jsonify({"error": "未找到PDF文件或没有第5页及以后的内容"}), 404
                
    except Exception as e:
        logging.error(f"分析PDF时出错: {e}")
        return jsonify({"error": str(e)}), 500

@search_bp.route('/ask_question', methods=['POST'])
def ask_question():
    try:
        question = request.form.get('question')
        if not question:
            return jsonify({"error": "未提供问题"}), 400
            
        answer = ask_siflow(question)
        if answer is None:
            return jsonify({"error": "无法获取答案"}), 500
            
        return jsonify({
            "success": True,
            "answer": answer
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@search_bp.route('/output_images/<path:filename>')
def serve_image(filename):
    try:
        full_path = os.path.join('output_images', filename)
        logging.info(f"图片请求: {full_path}")
        
        if os.path.exists(full_path):
            return send_file(full_path)
        else:
            base_name = os.path.splitext(filename)[0]
            match = re.match(r'(.+)_page_\d+', base_name)
            if match:
                folder_name = match.group(1)
                alt_path = os.path.join('output_images', folder_name, filename)
                logging.info(f"尝试替代路径: {alt_path}")
                if os.path.exists(alt_path):
                    return send_file(alt_path)
            
            logging.error(f"图片不存在: {full_path}")
            return jsonify({"error": "图片不存在"}), 404
            
    except Exception as e:
        logging.error(f"提供图片服务时出错: {e}")
        return jsonify({"error": str(e)}), 500

@search_bp.route('/debug/check_files')
def debug_check_files():
    try:
        result = {
            'output_images_exists': os.path.exists('output_images'),
            'output_images_content': [],
            'pdf_files_exists': os.path.exists('pdf_files'),
            'pdf_files_content': []
        }
        
        if os.path.exists('output_images'):
            result['output_images_content'] = os.listdir('output_images')
            if '12G614-1' in result['output_images_content']:
                folder_path = os.path.join('output_images', '12G614-1')
                result['12G614-1_content'] = os.listdir(folder_path)
                result['12G614-1_path'] = os.path.abspath(folder_path)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@search_bp.route('/debug/test_image/<int:page>')
def test_image(page):
    try:
        paths_to_try = [
            os.path.join('output_images', '12G614-1', f'12G614-1_page_{page}.png'),
            os.path.join('output_images', f'12G614-1_page_{page}.png'),
            os.path.join('output_images', '12G614-1', f'page_{page}.png')
        ]
        
        result = {
            'page': page,
            'attempts': []
        }
        
        for path in paths_to_try:
            exists = os.path.exists(path)
            result['attempts'].append({
                'path': path,
                'exists': exists,
                'absolute': os.path.abspath(path) if exists else None
            })
            if exists:
                return send_file(path)
        
        return jsonify(result), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@search_bp.route('/debug/pdf_info/<file_name>')
def debug_pdf_info(file_name):
    try:
        if not file_name.endswith('.pdf'):
            file_name += '.pdf'
            
        base_name = os.path.splitext(file_name)[0]
        
        result = {
            'file_name': file_name,
            'base_name': base_name,
            'database': [],
            'filesystem': []
        }
        
        try:
            with pymysql.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT page_num, image_path, created_at 
                        FROM pdf_info 
                        WHERE file_name = %s
                        ORDER BY page_num
                    """, (file_name,))
                    
                    db_results = cursor.fetchall()
                    for row in db_results:
                        result['database'].append({
                            'page_num': row[0],
                            'image_path': row[1],
                            'created_at': str(row[2])
                        })
        except Exception as e:
            result['database_error'] = str(e)
        
        folder_path = os.path.join(OUTPUT_FOLDER, base_name)
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            png_files = [f for f in files if f.endswith('.png')]
            
            for img_file in png_files:
                match = re.search(r'page_(\d+)', img_file)
                if match:
                    page_num = int(match.group(1))
                    file_path = os.path.join(folder_path, img_file)
                    result['filesystem'].append({
                        'page_num': page_num,
                        'image_path': img_file,
                        'exists': os.path.exists(file_path),
                        'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    })
            
            result['filesystem'].sort(key=lambda x: x['page_num'])
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def init_search():
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TEMP_FOLDER]:
        os.makedirs(folder, exist_ok=True)
        logging.info(f"确保目录存在: {folder}")
        
    if not init_database():
        logging.error("搜索功能初始化失败：数据库初始化错误")
        return False
        
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    if pdf_files:
        logging.info(f"找到PDF文件: {pdf_files}")
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                logging.info(f"开始处理PDF: {pdf_file}")
                process_pdf(pdf_file)
    else:
        logging.warning("没有找到PDF文件")
    
    return True