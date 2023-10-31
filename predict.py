# Imports necessary libraries
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from collections import OrderedDict
import json
from PIL import Image
import argparse

# Defines command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--json_file', type=str, default='cat_to_name.json', help='Path to JSON file with category names.')
parser.add_argument('--test_file', type=str, default='flowers/train/43/image_02364.jpg', help='Path to the image for prediction.')
parser.add_argument('--checkpoint_file', type=str, default='checkpoint.pth', help='Path to the checkpoint file for model loading.')
parser.add_argument('--topk', type=int, default=5, help='Number of top predictions to display.')
parser.add_argument('--gpu', default='gpu', type=str, help='Specify "gpu" to use GPU for prediction, default is "gpu".')

# Maps command-line arguments to variables
cl_inputs = parser.parse_args()
json_file = cl_inputs.json_file
test_file = cl_inputs.test_file
checkpoint_file = cl_inputs.checkpoint_file
topk = cl_inputs.topk
gpu = cl_inputs.gpu

# Loads category names from the JSON file
with open(json_file, 'r') as f:
    cat_to_name = json.load(f)

# Defines a function to load the model from a checkpoint file
def load_model(checkpoint_file=checkpoint_file):
    # Load model architecture, hyperparameters, and class to index mapping
    checkpoint = torch.load(checkpoint_file)
    arch = checkpoint['arch']
    lr = checkpoint['lr']
    hidden_layer = checkpoint['hidden_layer']
    gpu = checkpoint['gpu']
    epochs = checkpoint['epochs']
    dropout = checkpoint['dropout']
    classifier = checkpoint['classifier']
    state_dict = checkpoint['state_dict']
    class_to_idx = checkpoint['class_to_idx']

    # Load the pre-trained model based on the architecture specified in the checkpoint
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)

    # Configure the model's classifier, class to index mapping, and load its state_dict
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    model.load_state_dict(state_dict)

    # Set model parameters to not require gradients
    for param in model.parameters():
        param.requires_grad = False

    return model

# Build the model using the load_model function and the loaded checkpoint file
loaded_model = load_model()

# Defines a function to process an image
def process_image(image=test_file):
    '''Scales, crops, and normalizes a PIL image for a PyTorch model, returning a NumPy array'''
    picture = Image.open(image)

    transformation = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                     ])

    np_array = transformation(picture).float()

    return np_array

# Defines a function to predict the class of an image
def predict(image_path=test_file, model=loaded_model, topk=topk, gpu=gpu):
    # Process the image
    image = process_image(image_path)
    image = image.float().unsqueeze_(0)

    # Move to GPU if specified
    if gpu == 'gpu':
        model.to('cuda:0')

    # Create prediction scores
    with torch.no_grad():
        if gpu == 'gpu':
            image = image.to('cuda')

        output = model.forward(image)

    prediction = F.softmax(output.data, dim=1)

    # Get top k predictions
    probs, indices = prediction.topk(topk)
    probs = probs.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]

    # Map indices to class labels
    idx_to_class = {v:k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]

    return probs, classes

# Run the "predict" function on the test image and print the results
probs, classes = predict(test_file, loaded_model, topk)
print(probs)
print(classes)
