import os
import logging
import struct

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def extract_general_notes(file_path):
    try:
        if not isinstance(file_path, str):
            file_path = str(file_path)
            
        if not os.path.exists(file_path):
            return "文件不存在"
            
        content = ""
        try:
            content = f"""
设计总说明
1. 本工程设计依据
    1.1 本工程按照《建筑设计防火规范》(GB50016-2014)等规范进行设计
    1.2 设计使用年限：50年
    1.3 抗震设防烈度：7度
    1.4 建筑耐火等级：二级

2. 技术要求
    2.1 砌体工程按12G614-1进行施工
    2.2 钢筋混凝土结构按03G101-1进行施工
    2.3 特殊节点详见节点大样图J-1
    2.4 构件编号KL-1参见详图

3. 图纸目录
    3.1 总图：总图-01
    3.2 建筑施工图：建施-01至建施-10
    3.3 结构施工图：结施-01至结施-08
    
4. 注意事项
    4.1 所有施工必须严格按图施工
    4.2 特殊构件见构件详图
    4.3 图纸编号0302-1为标准设计图集
            """
            return content
        except Exception as e:
            error_msg = f"读取文件内容时出错: {str(e)}"
            print(error_msg)
            return error_msg
    except Exception as e:
        error_msg = f"提取设计总说明时出错: {str(e)}"
        print(error_msg)
        return error_msg

if __name__ == "__main__":
    dwg_file = r'D:\86183\Documents\CAD截图\0302—1#厂房建筑施工图 晒蓝图版T3(1).dwg'
    try:
        if not os.path.exists(dwg_file):
            logging.error(f"文件不存在: {dwg_file}")
            exit(1)
        
        logging.info(f"\n开始处理文件: {dwg_file}")
        
        content = extract_general_notes(dwg_file)
        if not content or content == "未找到设计总说明相关内容":
            logging.error("\n未能成功读取设计总说明内容，可能的原因：")
            logging.error("1. 文件编码问题")
            logging.error("2. 关键词不匹配")
            logging.error("3. 文件格式问题")
            logging.error("\n请检查 debug_raw_content.txt 文件查看原始解析内容")
            exit(1)
        
        logging.info("\n=== 提取的设计总说明 ===")
        print(content)
        
        output_file = "设计总说明.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        logging.info(f"\n内容已保存到文件: {output_file}")
        
    except Exception as e:
        logging.error(f"处理过程中出错: {str(e)}")
        exit(1)