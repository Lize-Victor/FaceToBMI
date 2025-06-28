import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 推荐使用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# 数据准备
# ------------------------------------------------------------
# 加载元数据 - 只读取需要的列
df = pd.read_csv('model/VIP_attribute/annotation.csv', usecols=['image', 'height', 'weight', 'BMI'])
print(f"数据集样本数: {len(df)}")
print(f"BMI范围: {df['BMI'].min():.1f} - {df['BMI'].max():.1f}")


# 数据预处理
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
image_dir = 'model/VIP_attribute/images'
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

    # 每处理100个样本打印一次进度
    if len(X) % 100 == 0:
        print(f"已处理 {len(X)} 个样本")


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


def create_improved_model():
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling=None
    )

    # 初始冻结基础层
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    bmi_output = layers.Dense(1, activation='linear', name='bmi')(x)

    model = models.Model(inputs, bmi_output)
    return model


X_train, X_val, y_bmi_train, y_bmi_val = train_test_split(
    X, y_bmi, test_size=0.2, random_state=42
)

# 第一阶段训练
model = create_improved_model()
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='mse',
    metrics=['mae']
)

# 修正的回调函数 - 使用正确的权重文件后缀
stage1_checkpoint = callbacks.ModelCheckpoint(
    'best_stage1.weights.h5',  # 使用 .weights.h5 后缀
    monitor='val_mae',
    save_best_only=True,
    save_weights_only=True,
    mode='min',
    verbose=1
)

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

print("第一阶段训练 - 冻结基础层...")
history1 = model.fit(
    datagen.flow(X_train, y_bmi_train, batch_size=32),
    epochs=30,
    validation_data=(X_val, y_bmi_val),
    callbacks=[early_stopping, reduce_lr, stage1_checkpoint]
)

# 第二阶段训练
print("\n第二阶段训练 - 解冻部分层...")

# 重新创建模型结构并加载最佳权重
model = create_improved_model()
model.load_weights('best_stage1.weights.h5')

# 解冻部分层 - 更安全的解冻策略
# 只解冻最后10%的层，避免解冻BatchNormalization层
total_layers = len(model.layers[1].layers)
for layer in model.layers[1].layers[-int(total_layers * 0.1):]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

# 重新编译
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='mse',
    metrics=['mae']
)

# 第二阶段回调 - 使用正确的权重文件后缀
final_checkpoint = callbacks.ModelCheckpoint(
    'best_final.weights.h5',  # 使用 .weights.h5 后缀
    monitor='val_mae',
    save_best_only=True,
    save_weights_only=True,
    mode='min',
    verbose=1
)

# 训练
print("开始第二阶段训练...")
history2 = model.fit(
    datagen.flow(X_train, y_bmi_train, batch_size=32),
    epochs=50,
    validation_data=(X_val, y_bmi_val),
    callbacks=[early_stopping, reduce_lr, final_checkpoint]
)

# 加载最佳权重并保存完整模型
model.load_weights('best_final.weights.h5')
model.save('bmi_prediction_model_1.keras')
    
print("模型训练完成并保存为 bmi_prediction_model")

# 评估模型
val_loss, val_mae = model.evaluate(X_val, y_bmi_val)
print(f"\n最终模型性能:")
print(f"验证集损失: {val_loss:.4f}")
print(f"验证集BMI MAE: {val_mae:.4f}")


# 使用Matplotlib绘制图像

# 1. 绘制训练历史曲线
def plot_training_history(history1, history2):
    """绘制两阶段的训练历史曲线"""
    plt.figure(figsize=(15, 10))

    # 合并两个历史记录
    epochs1 = len(history1.history['loss'])
    epochs2 = len(history2.history['loss'])
    total_epochs = epochs1 + epochs2

    # 创建合并的历史记录
    combined_history = {
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
        'mae': history1.history['mae'] + history2.history['mae'],
        'val_mae': history1.history['val_mae'] + history2.history['val_mae']
    }

    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(combined_history['loss'], label='训练损失')
    plt.plot(combined_history['val_loss'], label='验证损失')
    plt.axvline(x=epochs1, color='r', linestyle='--', alpha=0.3, label='第二阶段开始')
    plt.title('训练和验证损失')
    plt.ylabel('损失 (MSE)')
    plt.xlabel('轮次')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 绘制MAE曲线
    plt.subplot(2, 1, 2)
    plt.plot(combined_history['mae'], label='训练MAE')
    plt.plot(combined_history['val_mae'], label='验证MAE')
    plt.axvline(x=epochs1, color='r', linestyle='--', alpha=0.3, label='第二阶段开始')
    plt.title('训练和验证平均绝对误差 (MAE)')
    plt.ylabel('MAE')
    plt.xlabel('轮次')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()


# 2. 绘制预测值与真实值的散点图
def plot_predictions_vs_actuals(model, X_val, y_val):
    """绘制预测值与真实值的散点图"""
    # 进行预测
    y_pred = model.predict(X_val).flatten()

    plt.figure(figsize=(10, 8))

    # 计算线性回归线
    m, b = np.polyfit(y_val, y_pred, 1)
    reg_line = m * y_val + b

    # 绘制散点图和回归线
    plt.scatter(y_val, y_pred, alpha=0.6, edgecolors='w', s=60)
    plt.plot(y_val, reg_line, color='r', linestyle='--', linewidth=2)

    # 绘制对角线（理想预测线）
    min_val = min(min(y_val), min(y_pred))
    max_val = max(max(y_val), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k-', alpha=0.7)

    # 计算R²
    residuals = y_val - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # 添加统计信息
    plt.text(0.05, 0.95, f'MAE: {val_mae:.2f}\nR²: {r2:.3f}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title('预测BMI vs 真实BMI')
    plt.xlabel('真实BMI')
    plt.ylabel('预测BMI')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig('predictions_vs_actuals.png', dpi=300)
    plt.show()


# 3. 绘制误差分布直方图
def plot_error_distribution(model, X_val, y_val):
    """绘制预测误差分布直方图"""
    # 进行预测
    y_pred = model.predict(X_val).flatten()
    errors = y_pred - y_val

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')

    # 添加统计信息
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.axvline(mean_error, color='r', linestyle='dashed', linewidth=2,
                label=f'平均误差: {mean_error:.2f}')
    plt.axvline(mean_error + std_error, color='g', linestyle='dashed', linewidth=1)
    plt.axvline(mean_error - std_error, color='g', linestyle='dashed', linewidth=1,
                label=f'标准差: ±{std_error:.2f}')

    plt.title('预测误差分布')
    plt.xlabel('预测误差 (预测值 - 真实值)')
    plt.ylabel('样本数量')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig('error_distribution.png', dpi=300)
    plt.show()


# 4. 可视化一些预测结果
def visualize_predictions(model, X_val, y_val, num_samples=9):
    """可视化一些预测结果"""
    # 随机选择样本
    indices = np.random.choice(len(X_val), num_samples, replace=False)
    sample_images = X_val[indices]
    sample_true = y_val[indices]

    # 进行预测
    sample_pred = model.predict(sample_images).flatten()

    plt.figure(figsize=(15, 12))

    # 计算网格大小
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    for i, idx in enumerate(indices):
        plt.subplot(grid_size, grid_size, i + 1)

        # 显示图像 (需要将归一化的图像反归一化)
        img = sample_images[i] * 255.0
        img = img.astype(np.uint8)

        # 如果是BGR格式转换为RGB
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img)

        # 添加标题显示真实值和预测值
        error = sample_pred[i] - sample_true[i]
        title_color = 'red' if abs(error) > 3 else ('orange' if abs(error) > 2 else 'green')
        plt.title(f"真实: {sample_true[i]:.1f}\n预测: {sample_pred[i]:.1f}\n误差: {error:.1f}",
                  color=title_color, fontsize=10)
        plt.axis('off')

    plt.suptitle('样本预测结果 (红>3, 橙>2, 绿<2误差)', fontsize=16)
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300)
    plt.show()


# 调用绘图函数
print("\n绘制训练历史曲线...")
plot_training_history(history1, history2)

print("绘制预测值与真实值散点图...")
plot_predictions_vs_actuals(model, X_val, y_bmi_val)

print("绘制误差分布直方图...")
plot_error_distribution(model, X_val, y_bmi_val)

print("可视化一些预测结果...")
visualize_predictions(model, X_val, y_bmi_val, num_samples=9)

