import cv2
import numpy as np
import logging
from PIL import Image
import os

class ImageProcessor:
    def __init__(self):
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.kernel_sharpening = np.array([[-1,-1,-1],
                                         [-1, 9,-1],
                                         [-1,-1,-1]])

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            denoised = cv2.fastNlMeansDenoising(enhanced, 
                                               None, 
                                               h=10,
                                               templateWindowSize=7,
                                               searchWindowSize=21)
            
            sharpened = cv2.filter2D(denoised, -1, self.kernel_sharpening)
            
            _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = np.ones((2,2), np.uint8)
            morphology = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            result = cv2.cvtColor(morphology, cv2.COLOR_GRAY2BGR)
            
            self.logger.info("图像增强完成")
            return result
            
        except Exception as e:
            self.logger.error(f"图像增强失败: {str(e)}")
            return image

    def save_image(self, image: np.ndarray, output_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            cv2.imwrite(output_path, image)
            self.logger.info(f"图像已保存至: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存图像失败: {str(e)}")
            return False

    def process_engineering_drawing(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            edges = cv2.Canny(enhanced, 50, 150)
            
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                                  minLineLength=100, maxLineGap=10)
            
            line_image = np.zeros_like(image)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
            
            result = cv2.addWeighted(image, 0.8, line_image, 0.2, 0)
            
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced_lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            return result
            
        except Exception as e:
            print(f"工程图纸处理错误: {str(e)}")
            return image