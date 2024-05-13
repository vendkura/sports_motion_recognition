import torch
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from train_V2 import initialize_device, load_transforms, load_datasets, create_dataloaders, load_model
import numpy as np


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
    plt.savefig('output.png')
    plt.close()

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

    model_path = 'best_model_weights.h5'
    model = load_model(model_path, device)

    visualize_model(model, dataloaders, class_names, num_images=6, filename='model_predictions.png')
