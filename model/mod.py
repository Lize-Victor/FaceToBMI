import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei'] # 推荐使用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


# 1. 数据准备
# ------------------------------------------------------------
# 加载元数据 - 只读取需要的列
df = pd.read_csv('VIP_attribute/annotation.csv', usecols=['image', 'height', 'weight', 'BMI'])
print(f"数据集样本数: {len(df)}")
print(f"BMI范围: {df['BMI'].min():.1f} - {df['BMI'].max():.1f}")


# 2. 数据预处理
# ------------------------------------------------------------
# 人脸检测函数 - 使用OpenCV
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


# 创建图像和标签数组
X, y_bmi = [], []
skipped_count = 0

print("\n开始处理图像...")
image_dir = 'VIP_attribute/images'
if not os.path.exists(image_dir):
    print(f"错误: 图像目录不存在 {image_dir}")
    exit(1)

for idx, row in df.iterrows():
    # 构建图像路径
    img_name = row['image']
    img_path = os.path.join(image_dir, f"{img_name}.jpg")

    # 检查文件是否存在
    if not os.path.exists(img_path):
        print(f"警告: 文件不存在 {img_path}")
        skipped_count += 1
        continue

    # 检测人脸
    face_img = detect_and_crop_face(img_path, margin=0.3)
    if face_img is None:
        print(f"警告: 无法处理图像 {img_path}")
        skipped_count += 1
        continue

    X.append(face_img)
    y_bmi.append(row['BMI'])

    # 每处理10个样本打印一次进度
    if len(X) % 10 == 0:
        print(f"已处理 {len(X)} 个样本")

# 检查是否加载了样本
if len(X) == 0:
    print("错误: 没有加载任何图像!")
    print("请检查: ")
    print(f"1. 图像目录 '{image_dir}' 是否存在")
    print(f"2. CSV中的图像名称是否与目录中的.jpg文件匹配")
    exit(1)

# 转换为NumPy数组
X = np.array(X)
y_bmi = np.array(y_bmi)

# 归一化图像
if X.dtype != np.float32:
    X = X.astype(np.float32) / 255.0

print(f"\n预处理后的数据形状: {X.shape}")
print(f"BMI标签形状: {y_bmi.shape}")
print(f"跳过的样本数: {skipped_count}")

# 3. 数据增强
# ------------------------------------------------------------
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)


# 4. 构建BMI预测模型
# ------------------------------------------------------------
def create_bmi_model():
    # 使用EfficientNetB0作为基础模型
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    base_model.trainable = True  # 微调所有层

    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # BMI预测输出
    bmi_output = layers.Dense(1, activation='linear', name='bmi')(x)

    model = models.Model(inputs=inputs, outputs=bmi_output)
    return model


model = create_bmi_model()

# 5. 编译模型
# ------------------------------------------------------------
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='mse',
    metrics=['mae']
)

# 6. 训练模型
# ------------------------------------------------------------
# 划分训练集和验证集
X_train, X_val, y_bmi_train, y_bmi_val = train_test_split(
    X, y_bmi, test_size=0.2, random_state=42
)

# 打印形状验证
print("\n训练集形状验证:")
print(f"X_train: {X_train.shape}, y_bmi_train: {y_bmi_train.shape}")
print(f"验证集形状验证:")
print(f"X_val: {X_val.shape}, y_bmi_val: {y_bmi_val.shape}")

# 回调函数
early_stopping = callbacks.EarlyStopping(
    monitor='val_mae',
    patience=10,
    restore_best_weights=True,
    mode='min'
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6
)

# 训练模型
print("\n开始训练模型...")
history = model.fit(
    datagen.flow(X_train, y_bmi_train, batch_size=32),
    epochs=20,
    validation_data=(X_val, y_bmi_val),
    callbacks=[early_stopping, reduce_lr]
)

# 7. 评估模型
# ------------------------------------------------------------
# 保存为更通用的TensorFlow SavedModel格式
model.save('bmi_prediction_model', save_format='tf')

# 评估模型
val_loss, val_mae = model.evaluate(X_val, y_bmi_val)
print(f"\n验证集损失: {val_loss:.4f}")
print(f"验证集BMI MAE: {val_mae:.4f}")

# 可视化训练过程
plt.figure(figsize=(12, 6))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.ylabel('损失')
plt.xlabel('轮次')
plt.legend()

# MAE曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='训练MAE')
plt.plot(history.history['val_mae'], label='验证MAE')
plt.title('BMI平均绝对误差')
plt.ylabel('MAE')
plt.xlabel('轮次')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()


# 8. 进行预测
# ------------------------------------------------------------
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


# 示例预测
sample_image = 'VIP_attribute/images/f_001.jpg'
if os.path.exists(sample_image):
    bmi_pred = predict_bmi(sample_image)
    print(f"\n预测结果: BMI={bmi_pred:.2f}")
else:
    # 尝试使用数据集中的第一个图像
    if len(df) > 0:
        first_img = os.path.join(image_dir, f"{df.iloc[0]['image']}.jpg")
        if os.path.exists(first_img):
            bmi_pred = predict_bmi(first_img)
            actual_bmi = df.iloc[0]['BMI']
            print(f"\n示例预测: {first_img}")
            print(f"预测BMI: {bmi_pred:.2f}, 实际BMI: {actual_bmi:.2f}")
        else:
            print("\n无法找到示例图像进行预测")
    else:
        print("\n数据集为空，无法进行示例预测")