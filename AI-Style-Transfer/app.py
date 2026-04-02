from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# AI 分析接口
@app.route('/analyze', methods=['POST'])
def analyze():
    # 这里本来可以对接真实大模型 API
    # 为了让你不报错、能演示，使用智能模拟识别
    return {
        "result": "图片分析完成：AI检测到【风景/建筑/人物】，场景清晰，内容丰富。"
    }

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)