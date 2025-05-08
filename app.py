from flask import Flask, render_template, request
import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_frames(video_path, frame_rate=30):
    frames = []
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        count += 1
    cap.release()
    return frames[:10]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files['video']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            frames = extract_frames(filepath)
            results = []
            for frame in frames:
                img = transform(frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(img)
                    pred = torch.argmax(output, dim=1).item()
                    results.append(pred)
            real_pct = results.count(0) / len(results) * 100
            fake_pct = results.count(1) / len(results) * 100
            result = f"Real: {real_pct:.2f}%, Fake: {fake_pct:.2f}% — Prediction: {'Real' if real_pct > fake_pct else 'Fake'}"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
