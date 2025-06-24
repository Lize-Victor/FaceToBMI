import zmq


def run_backend(port=5555):
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # 使用 REP 模式
    socket.bind(f"tcp://localhost:{port}")  # 绑定到指定端口
    print(f"ZeroMQ 后端已启动，监听端口 {port} ...")
    while True:
        image_bytes = socket.recv()  # 接收图片数据
        print(f"收到图片，大小：{len(image_bytes)} 字节")
        # 这里可以保存图片或做进一步处理
        socket.send(b"successfully rec")  # 返回响应

if __name__ == "__main__":
    run_backend()