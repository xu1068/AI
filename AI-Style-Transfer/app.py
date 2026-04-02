from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)

# 加载前沿轻量化AI风格迁移模型
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 前端路由
@app.route('/')
def index():
    return render_template('index.html')

# AI风格迁移接口
@app.route('/transfer', methods=['POST'])
def transfer():
    # 获取参数（可调参）
    style_strength = float(request.form.get('strength', 0.8))
    image_data = request.form['image']
    
    # 图片预处理
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((384, 384))
    input_tensor = np.array(image, dtype=np.float32)[np.newaxis, ...] / 255.0

    # AI推理
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    output = np.clip(output * 255, 0, 255).astype(np.uint8)

    # 结果返回
    result_img = Image.fromarray(output)
    buffer = io.BytesIO()
    result_img.save(buffer, format='JPEG')
    result_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return jsonify({'result': f'data:image/jpeg;base64,{result_base64}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
