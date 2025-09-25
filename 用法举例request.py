from flask import Flask, request

app = Flask(__name__)

# 定义文件上传路由
@app.route('/upload', methods=['POST'])
def upload():
    # 检查请求中是否包含名为 'upload_file' 的文件
    if 'upload_file' not in request.files:
        return "没有找到上传的文件", 400
    file = request.files['upload_file']
    # 检查文件名是否为空
    if file.filename == '':
        return "文件名无效", 400
    # 这里简单打印文件名，实际应用中可以进行保存等操作
    print(f"接收到上传的文件，文件名是: {file.filename}")
    return "文件上传成功", 200

if __name__ == "__main__":
    app.run(debug=True)
