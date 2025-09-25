from flask import Flask, session, request

app = Flask(__name__)
app.secret_key = "dev-secret"  # 设置密钥

@app.route("/", methods=["GET", "POST"])
def index():
    # 从session中获取count，若不存在则设为0
    session["count"] = session.get("count", 0)
    if request.method == "POST":
        session["count"] += 1  # 每次POST请求，count加1
    return f"你点击了 {session['count']} 次<button method='post' form='f'>点我</button><form id='f' method='post'></form>"

if __name__ == "__main__":
    app.run(debug=True)
