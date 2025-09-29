from flask import Flask, render_template, request, redirect, url_for, flash
import hashlib
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 用于 flash 消息，需设置为随机字符串


# 密码加密函数
def encrypt_password(password):
    return hashlib.md5(password.encode()).hexdigest()


# 检查用户是否存在
def check_user_exist(username):
    if not os.path.exists("user_data.txt"):
        return False
    with open("user_data.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line.split("|")[0] == username:
                return True
    return False


# 登录页面路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        if not username or not password:
            flash('用户名和密码不能为空！')
            return render_template('login.html')

        if not check_user_exist(username):
            flash('用户名未注册！')
            return render_template('login.html')

        encrypted_pwd = encrypt_password(password)
        with open("user_data.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    stored_user, stored_pwd = line.split("|")
                    if stored_user == username and stored_pwd == encrypted_pwd:
                        flash('登录成功！即将跳转到文件上传页面')
                        # 登录成功，跳转到文件上传页面
                        return redirect(url_for('upload'))
            flash('密码错误！')
            return render_template('login.html')
    return render_template('login.html')


# 注册页面路由
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        confirm_pwd = request.form['confirm_pwd'].strip()

        if not username or not password or not confirm_pwd:
            flash('所有字段不能为空！')
            return render_template('register.html')

        if password != confirm_pwd:
            flash('两次密码不一致！')
            return render_template('register.html')

        if check_user_exist(username):
            flash('用户名已注册！')
            return render_template('register.html')

        encrypted_pwd = encrypt_password(password)
        with open("user_data.txt", "a", encoding="utf-8") as f:
            f.write(f"{username}|{encrypted_pwd}\n")

        flash('注册成功！请登录')
        return redirect(url_for('login'))
    return render_template('register.html')


# 文件上传页面路由
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # 这里可以添加文件上传的处理逻辑
        file = request.files['file']
        if file:
            # 假设将文件保存到 static/uploads 文件夹下
            file.save(os.path.join('static/uploads', file.filename))
            flash('文件上传成功！')
    return render_template('upload.html')


if __name__ == '__main__':
    # 确保上传文件夹存在
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
