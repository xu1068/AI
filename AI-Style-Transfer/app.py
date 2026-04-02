from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)

# 加载AI风格迁移双模型
predict_interpreter = tf.lite.Interpreter(model_path="style_predict.tflite")
transform_interpreter = tf.lite.Interpreter(model_path="style_transform.tflite")

predict_interpreter.allocate_tensors()
transform_interpreter.allocate_tensors()

# 图片预处理
def preprocess_image(image, size):
    image = image.convert('RGB').resize((size, size))
    return np.array(image, dtype=np.float32)[np.newaxis, ...] / 255.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transfer', methods=['POST'])
def transfer():
    try:
        # 获取参数
        style_strength = float(request.form.get('strength', 0.8))
        image_data = request.form['image']

        # 解码内容图
        image_bytes = base64.b64decode(image_data.split(',')[1])
        content_image = Image.open(io.BytesIO(image_bytes))
        content_tensor = preprocess_image(content_image, 384)

        # 加载风格图
        style_image = Image.open("style.jpg")
        style_tensor = preprocess_image(style_image, 256)

        # 预测风格特征
        pred_in = predict_interpreter.get_input_details()[0]['index']
        pred_out = predict_interpreter.get_output_details()[0]['index']
        predict_interpreter.set_tensor(pred_in, style_tensor)
        predict_interpreter.invoke()
        style_bottleneck = predict_interpreter.get_tensor(pred_out)

        # 调节风格强度
        style_bottleneck *= style_strength

        # 风格迁移
        trans_in = transform_interpreter.get_input_details()
        trans_out = transform_interpreter.get_output_details()[0]['index']

        transform_interpreter.set_tensor(trans_in[0]['index'], content_tensor)
        transform_interpreter.set_tensor(trans_in[1]['index'], style_bottleneck)
        transform_interpreter.invoke()

        # 结果处理
        output = transform_interpreter.get_tensor(trans_out)[0]
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        result_img = Image.fromarray(output)

        # 转base64
        buf = io.BytesIO()
        result_img.save(buf, format='JPEG')
        res_base64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            "code": 0,
            "result": f"data:image/jpeg;base64,{res_base64}"
        })

    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
