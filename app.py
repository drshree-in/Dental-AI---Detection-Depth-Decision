from flask import Flask, request
from flask_cors import CORS
import os
import base64
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from openai import OpenAI

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "*"}})

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

client = OpenAI()

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_with_gpt(image_path):
    print(image_path)
    base64_image = encode_image(image_path)
    resp = client.responses.create(
        model="gpt-4.1",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": (
                    "dont give any other respone then below mentioned: \n"
                    "Number of Teeth : \n"
                    "Name of the Teeth : \n"
                    "Density of the cavities in the Teeth : \n"
                    "Identify the tooth number according to FDI system. Check if the tooth has caries. \n"
                    "If present, check the depth of caries. \n"
                    "If caries involves the pulp, suggest root canal treatment \n"
                    "Identify the tooth, check if carious lesion is present, "
                    "if present see if pulpal involvement is there, if yes suggest root canal treatment :\n"
                    "What is the better Treatment for this : \n"
                    "healthy or unhealthy: "
                )},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
            ],
        }]
    )
    return resp.output_text


def manual_model_prediction():
    extraction = "/opt/tooth_analyzer/extraction/"
    rootcanal = "/opt/tooth_analyzer/rootcanal/"
    if not (os.path.isdir(extraction) and os.path.isdir(rootcanal)):
        return None, None  

    data, labels = [], []

    for file in os.listdir(extraction):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img = cv2.imread(os.path.join(extraction, file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            data.append(img.flatten()); labels.append(0)

    for file in os.listdir(rootcanal):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img = cv2.imread(os.path.join(rootcanal, file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            data.append(img.flatten()); labels.append(1)

    if not data:
        return None, None

    X = np.array(data); y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    acc = int(accuracy_score(y_test, model.predict(X_test)) * 100)
    return model, acc

def predict_image_with_manual_model(model, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    pred = model.predict(img.flatten().reshape(1, -1))[0]
    return "Extraction" if pred == 0 else "Root Canal Treatment"

@app.route('/', methods=['GET'])
def health():
    return "Tooth Analyzer API is running", 200

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if not file.filename:
        return "No selected file", 400 

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    ai_output = analyze_with_gpt(filepath)
    final_output = ai_output

    if "unhealthy" in ai_output.lower():
        model, acc = manual_model_prediction()
        if model is not None:
            manual_pred = predict_image_with_manual_model(model, filepath)
            final_output += f"\n\n🔬 Manual Model Accuracy: {acc}%\nManual Prediction: {manual_pred}"
        else:
            final_output += "\n\n⚠️ Manual model dataset not found. Skipping manual prediction."
    else:
        final_output += "\n\nEverything looks good — happy smile 😁"

    return final_output, 200

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=12355, debug=True)