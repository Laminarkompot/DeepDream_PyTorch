import torch
import matplotlib.pyplot as plt
import numpy as np
import nnio
import cv2
from torchvision import models

# Overwrite these values before launching script
PATH_TO_FILE = '/home/username/Downloads/picture.jpg'
SAVE_PATH = '/home/username/Downloads/picture_dream.png'

# Accelerate calculations
DEVICE = 'cuda'

# Number of iterations & learning rate
NUM_EPOCHS = 100
LR = 2

# Deep Dream adjustable parameters (try to toggle for better result)
ORIG_IMAGE_INTENSITY = 0.8
DREAM_INTENSITY = 0.09
RED_CH_FACTOR = 0.4
GREEN_CH_FACTOR = 0.3
BLUE_CH_FACTOR = 0.2
LAYER_3_CONTRIBUTION = 2
LAYER_4_CONTRIBUTION = 20

# Result picture dimensions (400x400 recommended)
H = 400
W = 400

# Define input picture preprocessing
preproc = nnio.Preprocessing(
    resize=(W, H),
    dtype='float32',
    divide_by_255=True,
    batch_dimension=True,
    channels_first=True
)

# Load and preprocess input picture
img = preproc(PATH_TO_FILE)

# Make a copy of input picture
orig_img = img

# Define forward hooks
hook_out3 = {}
def hook3(module, input, output):
    hook_out3['feats'] = output

hook_out4 = {}
def hook4(module, input, output):
    hook_out4['feats'] = output

# Create model
model = models.resnet18(pretrained=True)
model.requires_grad_ = False
model.to(DEVICE)

# Register forward hooks
model.layer3.register_forward_hook(hook3)
model.layer4.register_forward_hook(hook4)

for epoch in range(NUM_EPOCHS):

    img = torch.tensor(img, dtype=torch.float32, device=DEVICE, requires_grad=True)

    # Make forward pass
    out = model(img)

    # Get features
    feats3 = hook_out3['feats'][0, ::50, :, :]
    feats4 = hook_out4['feats'][0, ::100, :, :]

    # Compute loss
    loss = LAYER_4_CONTRIBUTION * (torch.mean(feats4 ** 2) ** 0.5) + LAYER_3_CONTRIBUTION * (torch.mean(feats3 ** 2) ** 0.5)

    # Compute gradients
    loss.backward()
    grad = img.grad
    img = img.detach().cpu().numpy()
    grad = grad.detach().cpu().numpy()

    # Normalize gradients
    grad = grad / (np.sqrt((grad**2).mean()) + 1e-05)
    
    # Gradient ascent step
    img = img + grad * LR

    print('epoch:', epoch, ' loss:', loss.detach().cpu().numpy())

# Adjust Deep Dream colours
img[:, 0, :, :] = img[:, 1, :, :] * RED_CH_FACTOR
img[:, 1, :, :] = img[:, 1, :, :] * GREEN_CH_FACTOR
img[:, 2, :, :] = img[:, 1, :, :] * BLUE_CH_FACTOR

# Unite the copy of initial picture with the Deep Dream picture
img = np.transpose(((DREAM_INTENSITY * img + ORIG_IMAGE_INTENSITY * orig_img) / 2).reshape(3, W, H), (1, 2, 0))

# Plotting
fig = plt.figure(figsize=(20, 10))
plt.imshow(img)
plt.axis('off')
plt.show()
fig.savefig(SAVE_PATH)

    

