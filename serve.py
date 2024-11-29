import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

onnx_model_path = "best.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")

        input_tensor = transform(img).unsqueeze(0)

        input_array = input_tensor.numpy()

        ort_inputs = {ort_session.get_inputs()[0].name: input_array}
        ort_outs = ort_session.run(None, ort_inputs)

        prediction = ort_outs[0].tolist()

        group1 = prediction[:10]
        group2 = prediction[10:]

        res1 = np.argmax(group1)
        res2 = np.argmax(group2)

        return jsonify({"predictions": [res1, res2]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
