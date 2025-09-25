import tkinter as tk
from tkinter import messagebox
import hashlib
import os
import webbrowser

global login_root
login_root = tk.Tk()

# ---------------------- 通用工具函数 ----------------------

login_root = None
register_root = None

def encrypt_password(password):
    """密码加密，统一用于注册和登录"""
    return hashlib.md5(password.encode()).hexdigest()


def check_user_exist(username):
    """检查用户名是否已注册"""
    if not os.path.exists("user_data.txt"):
        return False
    with open("user_data.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line.split("|")[0] == username:
                return True
    return False


# ---------------------- 1. 注册功能 ----------------------
def open_register_window():
    """打开注册窗口，注册成功后返回登录"""
    # 关闭登录窗口（若存在）
    global login_root,login_root
    try:
            # 尝试判断窗口是否存在（若已销毁，winfo_exists() 会触发异常）
        if 'login_root' in globals() and login_root.winfo_exists():
            login_root.destroy()
    except tk.TclError:
            # 捕获“窗口已销毁”的异常，直接跳过（说明窗口已经不存在了）
        pass

    register_root = tk.Tk()
    register_root.title("用户注册")
    register_root.geometry("400x300")
    register_root.resizable(False, False)

    # 注册逻辑
    def do_register():
        username = entry_reg_user.get().strip()
        password = entry_reg_pwd.get().strip()
        confirm_pwd = entry_reg_confirm.get().strip()

        # 表单验证
        if not username or not password or not confirm_pwd:
            messagebox.showerror("错误", "所有字段不能为空！")
            return
        if password != confirm_pwd:
            messagebox.showerror("错误", "两次密码不一致！")
            return
        if check_user_exist(username):
            messagebox.showerror("错误", "用户名已注册！")
            return

        # 存储用户数据
        encrypted_pwd = encrypt_password(password)
        with open("user_data.txt", "a", encoding="utf-8") as f:
            f.write(f"{username}|{encrypted_pwd}\n")

        messagebox.showinfo("成功", "注册成功！请登录")
        if register_root and register_root.winfo_exist():
            register_root.destroy()
        open_login_window()  # 注册后自动打开登录

    # 注册界面组件
    global entry_reg_user,entry_reg_pwd,entry_reg_confirm
    tk.Label(register_root, text="用户名：", font=("Arial", 12)).place(x=50, y=60)
    entry_reg_user = tk.Entry(register_root, font=("Arial", 12), width=20)
    entry_reg_user.place(x=130, y=60)

    tk.Label(register_root, text="密码：", font=("Arial", 12)).place(x=50, y=110)
    entry_reg_pwd = tk.Entry(register_root, font=("Arial", 12), width=20, show="*")
    entry_reg_pwd.place(x=130, y=110)

    tk.Label(register_root, text="确认密码：", font=("Arial", 12)).place(x=50, y=160)
    entry_reg_confirm = tk.Entry(register_root, font=("Arial", 12), width=20, show="*")
    entry_reg_confirm.place(x=130, y=160)

    tk.Button(register_root, text="注册", font=("Arial", 12), width=15, command=do_register).place(x=130, y=210)

    register_root.mainloop()


# ---------------------- 2. 登录功能（登录成功跳转网页） ----------------------
def open_login_window():
    """打开登录窗口，验证通过后跳转网页"""
    global login_root
    login_root = tk.Tk()
    login_root.title("用户登录")
    login_root.geometry("400x250")

    # 登录逻辑
    def do_login():
        username = entry_login_user.get().strip()
        password = entry_login_pwd.get().strip()

        # 表单验证
        if not username or not password:
            messagebox.showerror("错误", "用户名和密码不能为空！")
            return
        if not check_user_exist(username):
            messagebox.showerror("错误", "用户名未注册！")
            return

        # 密码验证
        encrypted_pwd = encrypt_password(password)
        with open("user_data.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    stored_user, stored_pwd = line.split("|")
                    if stored_user == username and stored_pwd == encrypted_pwd:
                        messagebox.showinfo("成功", "登录成功！即将打开网页")
                        # 跳转目标网页（无需改网页，直接用浏览器打开）
                        webbrowser.open("http://127.0.0.1:5000")  # 这里替换成你的目标网页
                        login_root.destroy()
                        return
        messagebox.showerror("错误", "密码错误！")

    # 登录界面组件
    tk.Label(login_root, text="用户名：", font=("Arial", 12)).place(x=60, y=70)
    entry_login_user = tk.Entry(login_root, font=("Arial", 12), width=20)
    entry_login_user.place(x=140, y=70)

    tk.Label(login_root, text="密码：", font=("Arial", 12)).place(x=60, y=120)
    entry_login_pwd = tk.Entry(login_root, font=("Arial", 12), width=20, show="*")
    entry_login_pwd.place(x=140, y=120)

    # 登录按钮
    tk.Button(login_root, text="登录", font=("Arial", 12), width=15, command=do_login).place(x=140, y=170)
    # 注册入口按钮
    tk.Button(login_root, text="还没账号？去注册", font=("Arial", 10),
              command=lambda: [login_root.destroy(), open_register_window()]).place(x=150, y=210)

    login_root.mainloop()


# ---------------------- 程序入口 ----------------------
if __name__ == "__main__":
    # 启动时直接打开登录窗口（未注册可点击按钮跳转注册）
    open_login_window()
