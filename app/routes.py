from flask import Blueprint
from flask import render_template, request, redirect, url_for, flash
import zmq
import os
bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/upload', methods=['POST'])
def upload():
    image = request.files.get('image') 
    image_url = None
    if image:        
        # 用绝对路径保存图片到 app/static/uploads
        upload_folder = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        image_path = os.path.join(upload_folder, image.filename)
        image.save(image_path)
        image_url = url_for('static', filename=f'uploads/{image.filename}')
        # 发送图片到后端
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:5555")
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        socket.send(image_bytes)
        # 接收后端响应
        backend_response = socket.recv().decode('utf-8')
        # 提取 bmi 数值
        if 'bmi:' in backend_response:
            bmi_str = backend_response.split('bmi:')[-1].strip()
            backend_response = round(float(bmi_str), 2)
        else:
            backend_response = '后端响应异常'
        result = f'{backend_response}'
    else:
        result = "No image uploaded."
    return render_template('index.html', result=result, image_url=image_url)