import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from my_dataset import MyDataset  # replace with your dataset class
from my_model import MyModel  # replace with your model class

# Define your augmentation pipeline
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(30),
])

# Load your model
model = MyModel().to('cuda')  # Assuming you are using GPU
model.load_state_dict(torch.load('path_to_your_model_weights.pth'))
model.eval()

# Load your dataset
# Example: creating a dataset instance
dataset = MyDataset('path_to_your_dataset')

def inference_with_tta(dataset, model, num_augments=8):
    preds = []
    for i in range(num_augments):
        augmented_dataset = MyDataset('path_to_your_dataset', transform=augmentation)
        dataloader = DataLoader(augmented_dataset, batch_size=32, shuffle=False)
        temp_preds = []
        with torch.no_grad():
            for images in dataloader:
                images = images.to('cuda')
                outputs = model(images)
                temp_preds.append(outputs.cpu().numpy())
        preds.append(np.concatenate(temp_preds, axis=0))
    return np.mean(preds, axis=0)  # Ensemble predictions by averaging

# Call your inference function
final_predictions = inference_with_tta(dataset, model)

# You can save your predictions or further process them
