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
import matplotlib.pyplot as plt
from PIL import Image
from tempfile import TemporaryDirectory

# Configuration and Initialization
def initialize_device():
    torch.backends.cudnn.benchmark = True
    plt.ion()   # interactive mode
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_transforms():
    # Data augmentation and normalization for training
    # Just normalization for validation
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
    print("Initializing, pre-treament of Datasets and Dataloaders...")
    return data_transforms

def load_datasets(data_dir, data_transforms):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    print("Loaded datasets.")
    return image_datasets

def create_dataloaders(image_datasets):
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    print("Created dataloaders [VAL][TRAIN].")
    return dataloaders


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, dataset_sizes, num_epochs=25):
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # torch.save({
            # 'epoch': epoch,
            # 'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # # 'loss': loss,
            # }, f'checkpoint_epoch_{epoch}.h5')

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(torch.load(best_model_params_path))
        ###################
        torch.save(model.state_dict(), 'best_model_weights.h5')
        print(f'Best model saved to best_model_weights.h5')
    return model
def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.savefig('output.png')  # Save the figure to a file
    plt.close()  # Close the plot to free up resources
    print("Image saved to output.png")

#### 
def visualize_model(model, num_images=6, filename='visualization_output.png'):
    """Function to visualize predictions and save the result as an image."""
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(10, 10))  # Define figure size

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])  # This uses imshow to display image

                if images_so_far == num_images:
                    plt.savefig(filename)  # Save the complete figure to a file
                    plt.close(fig)  # Close the figure to free up memory
                    model.train(mode=was_training)
                    return
        plt.savefig(filename)  # Ensure to save even if not enough images are found
        plt.close(fig)
        model.train(mode=was_training)



if __name__ == '__main__':
    device = initialize_device()
    data_transforms = load_transforms()
    data_dir = 'data/'
    image_datasets = load_datasets(data_dir, data_transforms)
    dataloaders = create_dataloaders(image_datasets)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    # Define the model, criterion, optimizer, and scheduler here
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Assuming 2 classes
    model = model.to(device)

    class_names = image_datasets['train'].classes

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    model = train_model(model, criterion, optimizer, scheduler, dataloaders, device, dataset_sizes, num_epochs=5)

    # Example visualization
    inputs, classes = next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[image_datasets['train'].classes[x] for x in classes])
     # Visualize some predictions and save the results
    visualize_model(model, num_images=6, filename='model_predictions.png')