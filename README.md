 Multimodal Psychodermatological Disorder Detection

This project is an **AI/ML system** that predicts **psychodermatological disorders** (stress, anxiety, depression) by combining **skin images** and **psychological text input** using a **multimodal deep learning model**.

---

## **Features**

- **CNN (ResNet18)** for skin image classification.
- **LSTM-based text classifier** for psychological data.
- **Multimodal fusion** combining both image and text features.
- Safe loading of **pretrained model checkpoint**.
- Simple **Flask web app** for user interaction.
- No separate templates or static folders needed (single-file app).

---

## **Project Structure**

Rayeesa_MLIntern_Assignment/
│
├── app.py # Main Flask app (single-file)
├── Data/
│ └── multimodal_model.pth # Pretrained model checkpoint
├── README.md
└── venv/ # Virtual environment (optional)

yaml
Copy code

---

## **Setup Instructions**

1. **Clone the repository**

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd Rayeesa_MLIntern_Assignment
Create virtual environment

bash
Copy code
python -m venv venv
Activate virtual environment

Windows (PowerShell):

powershell
Copy code
.\venv\Scripts\Activate.ps1
Windows (cmd):

cmd
Copy code
venv\Scripts\activate
Linux/Mac:

bash
Copy code
source venv/bin/activate
Install dependencies

bash
Copy code
pip install torch torchvision flask pillow
Run the Flask app

bash
Copy code
python app.py
Open browser at:

cpp
Copy code
http://127.0.0.1:5000/
Upload a skin image + enter psychological text → view predicted disorder.

Model Checkpoint
Path: Data/multimodal_model.pth

Includes CNN and LSTM weights.

Compatible with CNNModel, TextModel, and MultimodalModel defined in app.py.

How It Works
Image preprocessing: resize → tensor conversion.

Text preprocessing: tokenization → integer indices.

CNN + LSTM forward pass → combine features.

Fully connected layer → prediction (Stress / Anxiety / Depression).

License
This project is for educational/internship purposes.

yaml
Copy code

---

✅ **Next Steps**

1. Place `app.py` and `README.md` in your repository folder.  
2. Push to GitHub:

```bash
git init
git add .
git commit -m "Initial commit: Multimodal Psychodermatological Disorder Detection"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
