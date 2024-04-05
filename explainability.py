import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import torchvision
from tqdm import tqdm

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
EVAL_MODEL = 'result/max_acc.pth'  # Replace with the path to your model's weights
num_classes = 196
model = torchvision.models.resnet50(pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load(EVAL_MODEL, map_location=device))
model = model.to(device)
model.eval()

# Function to preprocess the input image
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    return img_tensor, img.size


# Function for Grad-CAM
def grad_cam(input_img, model, feature_layer):
    model.eval()
    features = []
    gradients = []

    def features_hook(module, input, output):
        features.append(output.cpu().data.numpy())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].cpu().data.numpy())

    feature_layer.register_forward_hook(features_hook)
    feature_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_img)
    _, preds = torch.max(output, 1)
    one_hot_output = torch.zeros_like(output)
    one_hot_output[0][preds] = 1

    # Backward pass
    model.zero_grad()
    output.backward(gradient=one_hot_output)

    # Get the captured features and gradients
    feature_output = features[0][0]
    gradient_output = gradients[0][0]

    # Pool the gradients across the channels
    pooled_gradients = np.mean(gradient_output, axis=(1, 2))

    # Weight the channels by the pooled gradients
    for i in range(pooled_gradients.shape[0]):
        feature_output[i, :, :] *= pooled_gradients[i]

    # The channel-wise mean of the resulting feature maps is our heatmap
    cam = np.mean(feature_output, axis=0)

    cam = np.maximum(cam, 0)  # Apply ReLU
    cam = cv2.resize(cam, (input_img.shape[3], input_img.shape[2]))  # Resize to input image size
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam


# Create a directory to save the overlay images
overlay_dir = 'overlay_images'
if not os.path.exists(overlay_dir):
    os.makedirs(overlay_dir)

# Process images in the validation set
validation_dir = '/media/cvpr/CM_1/datasets/original_dataset/cars_dMix/val'  # Replace with your path
for root, dirs, files in tqdm(os.walk(validation_dir), desc="Processing folders"):
    for file in tqdm(files, desc="Processing images"):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(root, file)
            input_img, original_size = preprocess_image(img_path)

            # Generate Grad-CAM
            cam = grad_cam(input_img, model, model.layer4[2].conv3)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

            # Read and resize the original image
            original_img = cv2.imread(img_path, 1)
            original_img = cv2.resize(original_img, (heatmap.shape[1], heatmap.shape[0]))

            # Overlay heatmap onto original image
            overlay_img = cv2.addWeighted(original_img, 0.5, heatmap, 0.5, 0)

            # Save the overlay image
            overlay_path = os.path.join(overlay_dir, f'overlay_{file}')
            cv2.imwrite(overlay_path, overlay_img)

print("Processing complete.")
