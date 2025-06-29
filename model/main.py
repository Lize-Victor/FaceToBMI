import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.losses import MeanSquaredError

# 加载模型时明确指定自定义对象
model = tf.keras.models.load_model('model/bmi_prediction_model.keras')

# 使用与训练时相同的人脸检测函数
def detect_and_crop_face(image_path, margin=0.3):
    """检测人脸并扩展边距"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"警告: 无法读取图像 {image_path}")
        return None

    # 检查图像是否成功加载
    if img.size == 0:
        print(f"警告: 空图像 {image_path}")
        return None

    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        # 取最大的人脸
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # 计算扩展边距
        margin_x = int(w * margin)
        margin_y = int(h * margin)

        # 应用边距并确保在图像范围内
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img.shape[1], x + w + margin_x)
        y2 = min(img.shape[0], y + h + margin_y)

        face_img = img[y1:y2, x1:x2]
        return cv2.resize(face_img, (224, 224))
    else:
        # 如果未检测到人脸，返回中心裁剪
        height, width = img.shape[:2]
        size = min(height, width)
        start_y = (height - size) // 2
        start_x = (width - size) // 2
        cropped = img[start_y:start_y + size, start_x:start_x + size]
        return cv2.resize(cropped, (224, 224))


def predict_bmi(image_path):
    """预测单张图像的BMI"""
    face_img = detect_and_crop_face(image_path, margin=0.3)
    if face_img is None:
        return None

    face_img = np.expand_dims(face_img, axis=0)
    if face_img.dtype != np.float32:
        face_img = face_img.astype(np.float32) / 255.0

    bmi_pred = model.predict(face_img)[0][0]
    return bmi_pred


# 使用示例
image_path = "1.png"  # 替换为你的图像路径
predicted_bmi = predict_bmi(image_path)
if predicted_bmi is not None:
    print(f"预测BMI: {predicted_bmi:.2f}")
else:
    print("无法处理该图像")