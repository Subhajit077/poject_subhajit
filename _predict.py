import os
import torch
from torchvision import transforms
from PIL import Image
from _model import TrafficSignCNN
from _config import resize_x, resize_y, num_classes

class TrafficSignPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrafficSignCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((resize_x, resize_y)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.class_names = [
            "Green Light", "Red Light", "Stop", 
            "Speed Limit 10", "Speed Limit 20", "Speed Limit 30",
            "Speed Limit 40", "Speed Limit 50", "Speed Limit 60",
            "Speed Limit 70", "Speed Limit 80", "Speed Limit 90",
            "Speed Limit 100", "Speed Limit 110", "Speed Limit 120"
        ]
    
    def predict_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs.data, 1)
            predicted_class = predicted.item()
        
        return self.class_names[predicted_class]
    
    def predict_directory(self, directory_path):
        predictions = {}
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                predictions[filename] = self.predict_image(file_path)
        return predictions

def classify_traffic_signs(input_path):
    predictor = TrafficSignPredictor("_checkpoints/final_weights.pth")
    if os.path.isdir(input_path):
        return predictor.predict_directory(input_path)
    elif os.path.isfile(input_path):
        return [predictor.predict_image(input_path)]
    else:
        raise ValueError("Invalid input path. Provide a valid file or directory path.")

if __name__ == "__main__":
    # Hardcoded input path
    input_path = "./data/test"  # Replace with your desired path

    # Call the classify_traffic_signs function
    try:
        results = classify_traffic_signs(input_path)
        if isinstance(results, dict):
            print("Predictions for directory:")
            for filename, prediction in results.items():
                print(f"{filename}: {prediction}")
        else:
            print("Prediction for image:")
            print(results[0])
    except ValueError as e:
        print(e)