AI赋能的工程构件识别与图纸流程一体化服务平台
简介
  本项目运用图像识别和人工智能技术，支持用户上传构件图片自动识别构件类型与相关信息；同时可对施工图进行智能剖分，将CAD工图中的不同内容整合为分析报告，实现基于编号页码的快速图片检索。
功能
（1）构件智能识别：基于AIA-YOLOv8模型实现建筑构件精准检测，自动识别梁、柱、板、墙等核心构件，输出构件类型及置信度等关键信息；
（2）工图智能剖分与检索：将页码与图像索引关联并建立检索系统，实现基于编号页码的快速图片搜索；
（3）工图智能梳理：通过DeepSeek-v3大模型实现建筑工图的语义解析，从而对查询的流程图进行实时解析；
（4）工图智能解析：将CAD工图以编码形式读取，通过DeepSeek-v3大模型理解编码，实现该工图的重点分析报告。
技术架构
前端：HTML5、CSS3、JavaScript、Font Awesome
后端：Flask、Python
AI模型：YOLOv8、PyTorch、Ultralytics
数据库：MySQL
文档处理：PyMuPDF
API服务：DeepSeek API
环境要求
Python 3.8 及以上版本
运行项目
python app.py
访问地址
首页：http://127.0.0.1:5000
构件识别：http://127.0.0.1:5000/recognition
图纸检索：http://127.0.0.1:5000/search
图纸分析：http://127.0.0.1:5000/analyze
配置说明
API 密钥配置：在 analyze.py 中配置相关 API 密钥
数据库配置：在 search.py 中配置 MySQL 连接
