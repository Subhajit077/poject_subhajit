import os
import torch
from torchvision import transforms
from PIL import Image
from _model import TrafficSignCNN
from _config import resize_x, resize_y

def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")
    
    model = TrafficSignCNN().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    model.eval()
    return model

def predict_image(model, image_path, transform, device, class_names):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        return class_names[predicted.item()]

def predict_directory(directory_path, model, transform, device, class_names):
    predictions = {}
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            predictions[filename] = predict_image(model, file_path, transform, device, class_names)
    return predictions

if __name__ == "__main__":
    # Hardcoded input path
    input_path = "./_data/test/"  # Replace with your desired path
    model_path = "./_checkpoints/final_weights.pth"  # Path to the model checkpoint

    # Define device and transformations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    class_names = [
        "Green Light", "Red Light", "Stop", 
        "Speed Limit 10", "Speed Limit 20", "Speed Limit 30",
        "Speed Limit 40", "Speed Limit 50", "Speed Limit 60",
        "Speed Limit 70", "Speed Limit 80", "Speed Limit 90",
        "Speed Limit 100", "Speed Limit 110", "Speed Limit 120"
    ]

    # Load the model
    model = load_model(model_path, device)

    # Predict for all images in the directory
    predictions = predict_directory(input_path, model, transform, device, class_names)

    # Print predictions
    print("Predictions:")
    for filename, prediction in predictions.items():
        print(f"{filename}: {prediction}")
