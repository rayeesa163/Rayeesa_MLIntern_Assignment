from flask import Flask, request, render_template_string
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# -------------------------------
# Define Models
# -------------------------------
class CNNModel(nn.Module):
    def __init__(self, out_features=128):
        super(CNNModel, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, out_features)
    def forward(self, x):
        return self.resnet(x)

class TextModel(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=50, hidden_dim=128):
        super(TextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 128)
    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.rnn(x)
        return self.fc(h_n[-1])

class MultimodalModel(nn.Module):
    def __init__(self, cnn_model, text_model, num_classes=3):
        super(MultimodalModel, self).__init__()
        self.cnn_model = cnn_model
        self.text_model = text_model
        self.fc = nn.Linear(128+128, num_classes)
    def forward(self, img, text):
        img_feat = self.cnn_model(img)
        text_feat = self.text_model(text)
        combined = torch.cat((img_feat, text_feat), dim=1)
        return self.fc(combined)

# -------------------------------
# Dummy word2idx and labels
# -------------------------------
word2idx = {"i":2, "am":3, "stressed":4, "anxious":5, "happy":6}
label_dict = {0:"Stress", 1:"Anxiety", 2:"Depression"}

# -------------------------------
# Preprocessing functions
# -------------------------------
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)

def preprocess_text(text):
    tokens = text.lower().split()
    idxs = [word2idx.get(tok,1) for tok in tokens]  # 1=unknown
    return torch.tensor([idxs])

# -------------------------------
# Load model
# -------------------------------
cnn = CNNModel()
text_model = TextModel()
model = MultimodalModel(cnn, text_model)
# Load pretrained checkpoint if exists
checkpoint_path = "Data/multimodal_model.pth"
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_dict = model.state_dict()
    filtered_dict = {k:v for k,v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    print(f"Loaded {len(filtered_dict)} compatible weights from checkpoint.")
except Exception as e:
    print("Checkpoint not loaded:", e)
model.eval()

# -------------------------------
# Flask App
# -------------------------------
app = Flask(__name__)

# HTML templates
INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Psychodermatology Detection</title>
</head>
<body>
    <h1>Psychodermatology Disorder Detection</h1>
    <form method="POST" enctype="multipart/form-data">
        <label>Select Skin Image:</label>
        <input type="file" name="image" required><br><br>
        <label>Enter Psychological Text:</label>
        <input type="text" name="text_input" placeholder="I feel stressed..." required><br><br>
        <input type="submit" value="Predict">
    </form>
    {% if error %}
        <p style="color:red">{{ error }}</p>
    {% endif %}
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
</head>
<body>
    <h1>Prediction Result</h1>
    <p><strong>Text Input:</strong> {{ text_input }}</p>
    <p><strong>Predicted Disorder:</strong> {{ prediction }}</p>
    <a href="/">Go Back</a>
</body>
</html>
"""

@app.route('/', methods=['GET','POST'])
def index():
    error = None
    if request.method=='POST':
        file = request.files.get('image')
        text_input = request.form.get('text_input')
        if not file or not text_input:
            error = "Please upload an image and enter text."
            return render_template_string(INDEX_HTML, error=error)
        # Save temporary image
        filepath = "temp.jpg"
        file.save(filepath)
        # Preprocess
        img_tensor = preprocess_image(filepath)
        text_tensor = preprocess_text(text_input)
        # Predict
        with torch.no_grad():
            output = model(img_tensor, text_tensor)
            pred_label = torch.argmax(output, dim=1).item()
            prediction = label_dict.get(pred_label,"Unknown")
        return render_template_string(RESULT_HTML, text_input=text_input, prediction=prediction)
    return render_template_string(INDEX_HTML, error=error)

if __name__ == "__main__":
    app.run(debug=True)
