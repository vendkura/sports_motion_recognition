# Imports
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
from PIL import Image
from tempfile import TemporaryDirectory

# Configuration and Initialization
def initialize_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_transforms():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def load_datasets(data_dir, data_transforms):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    return image_datasets

def create_dataloaders(image_datasets):
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    return dataloaders

def load_model(model_path, device):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def visualize_model(model, dataloaders, class_names, num_images=6, filename='visualization_output.png'):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                if images_so_far < num_images:
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted: {class_names[preds[j]]}')
                    imshow(inputs.cpu().data[j])
                else:
                    break

            if images_so_far == num_images:
                break

        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        model.train(mode=was_training)

    print(f"Visualizations saved to {filename}")

if __name__ == '__main__':
    device = initialize_device()
    data_transforms = load_transforms()
    data_dir = 'data/'
    image_datasets = load_datasets(data_dir, data_transforms)
    dataloaders = create_dataloaders(image_datasets)
    class_names = image_datasets['train'].classes

    # Load the pre-saved model
    model_path = 'best_model_weights.h5'
    model = load_model(model_path, device)

    # Visualize predictions
    visualize_model(model, dataloaders, class_names, num_images=6, filename='model_predictions.png')
