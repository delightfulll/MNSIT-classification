# MNIST Digit Classifier

A full-stack deep learning application that classifies handwritten digits (0–9). Upload a photo of a handwritten digit and the model predicts what it is in real time.

---

## Demo

1. Write a digit (0–9) on paper and take a photo
2. Upload the image on the web app
3. The neural network returns its prediction instantly

---

## Tech Stack

### Machine Learning
| Tool | Purpose |
|---|---|
| **PyTorch** | Neural network framework |
| **torchvision** | Image transforms and MNIST dataset |
| **NumPy** | Data manipulation and Otsu thresholding |
| **Pillow (PIL)** | Image loading and preprocessing |
| **Matplotlib** | Visualizing training results (notebook) |

### Backend
| Tool | Purpose |
|---|---|
| **FastAPI** | REST API server |
| **Uvicorn** | ASGI server to run FastAPI |
| **python-multipart** | Handling file uploads |

### Frontend
| Tool | Purpose |
|---|---|
| **Next.js 16** | React framework |
| **React 19** | UI library |
| **Tailwind CSS v4** | Styling |
| **TypeScript** | Type safety |

---

## Project Structure

```
MNIST Classifcation Project/
├── models/
│   └── MNIST_model.pth         # Trained model weights
├── notebook/
│   └── mnist_analysis.ipynb    # Training and analysis notebook
├── src/
│   ├── model.py                # Neural network architecture
│   ├── predict.py              # Inference logic
│   ├── datasets.py             # Data loading and splitting
│   ├── train.py                # Training loop
│   ├── validate.py             # Validation loop
│   └── test.py                 # Evaluation and per-class accuracy
├── api/
│   └── app.py                  # FastAPI backend
├── frontend/
│   └── app/
│       ├── page.tsx            # Main upload UI
│       ├── layout.tsx          # Root layout
│       └── globals.css         # Global styles
└── requirements.txt
```

---

## Model Architecture

A feedforward neural network (MLP) trained on the MNIST dataset (70,000 handwritten digit images).

```
Input (784)  →  Hidden (128, ReLU, Dropout 30%)  →  Hidden (64, ReLU, Dropout 30%)  →  Output (10, LogSoftmax)
```

- **Loss function:** NLLLoss
- **Optimizer:** Adam (lr=0.01)
- **Epochs:** 10
- **Test accuracy:** ~94.7%

---

## Setup & Running

### 1. Clone the repo and install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API

```bash
python -m api.app
```

The API runs at `http://localhost:8000`.

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

The app runs at `http://localhost:3000`.

---

## How It Works

1. You upload an image on the frontend
2. Next.js sends it to the FastAPI `/predict` endpoint
3. The image is preprocessed:
   - Converted to grayscale
   - Binarized using Otsu thresholding (handles any lighting/paper color)
   - Cropped tightly around the digit
   - Resized to 28×28 and inverted to match MNIST format
4. The neural network outputs probabilities for digits 0–9
5. The highest probability digit is returned and displayed

---

## Tips for Best Results

- Write the digit **large** and **centered** on a plain white or light-colored background
- Use a **dark pen** with clear contrast
- Avoid extra marks or shadows around the digit
