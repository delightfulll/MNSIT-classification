import io
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from src.model import Network

model = Network()
model.load_state_dict(torch.load("models/MNIST_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),  # invert to match MNIST (white digit, black background)
])


#makes it so the image is black on white and can accept it
def otsu_threshold(arr: np.ndarray) -> int:
    """Find the optimal ink/paper threshold automatically for any lighting condition."""
    counts = np.bincount(arr.flatten(), minlength=256)
    total = arr.size
    total_sum = float(np.dot(np.arange(256), counts))
    sum_b, weight_b, max_var, threshold = 0.0, 0, 0.0, 0
    for t in range(256):
        weight_b += counts[t]
        if weight_b == 0 or weight_b == total:
            continue
        weight_f = total - weight_b
        sum_b += t * counts[t]
        mean_b = sum_b / weight_b
        mean_f = (total_sum - sum_b) / weight_f
        var = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var > max_var:
            max_var = var
            threshold = t
    return threshold



#preprocesses the image to be a 28 x 28 image with a black background and a white digit
def preprocess(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    gray = gray.filter(ImageFilter.GaussianBlur(radius=1))
    arr = np.array(gray)

    t = otsu_threshold(arr)
    binary = np.where(arr <= t, 0, 255).astype(np.uint8)
    binary_img = Image.fromarray(binary)

    # Crop to bounding box of ink pixels
    mask = binary == 0
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any():
        return binary_img.resize((28, 28), Image.LANCZOS)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h, w = binary.shape
    pad = max(4, int(max(h, w) * 0.08))
    rmin, rmax = max(0, rmin - pad), min(h, rmax + pad)
    cmin, cmax = max(0, cmin - pad), min(w, cmax + pad)

    cropped = binary_img.crop((cmin, rmin, cmax, rmax))

    # Center on a square white canvas (preserves aspect ratio)
    side = max(cropped.width, cropped.height)
    square = Image.new("L", (side, side), 255)
    square.paste(cropped, ((side - cropped.width) // 2, (side - cropped.height) // 2))

    return square.resize((28, 28), Image.LANCZOS)


def predict(file_bytes: bytes) -> int:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    processed = preprocess(image)
    tensor = transform(processed).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)


    #print the probabilities for each digit
    probabilities = torch.exp(output)[0]
    print("\n--- Prediction probabilities ---")
    for i, p in enumerate(probabilities):
        print(f"  {i}: {p.item():.4f}")
    predicted_digit = torch.argmax(probabilities).item()
    print(f"  → Predicted: {predicted_digit}\n")
    return int(predicted_digit)
