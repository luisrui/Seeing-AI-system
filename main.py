from utils.CreateMaskData import CreateData
import cv2 as cv
import os
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms
import numpy as np
import os
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from facenet_pytorch import InceptionResnetV1
from torch import optim


def bad_check_prediction():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet = (
        InceptionResnetV1(classify=True, pretrained="vggface2", num_classes=50)
        .eval()
        .to(device)
    )
    resnet.load_state_dict(
        torch.load("model_parameter0.pth", map_location="cuda:0"), strict=False
    )
    preprocessing = transforms.Compose(
        [
            np.float32,
            transforms.ToTensor(),
            transforms.Resize((160, 160)),
            fixed_image_standardization,
        ]
    )
    testset = CustomImageDataset(
        csv_file="test.csv", root_dir="crop-mask-face/test/", transform=preprocessing
    )
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
    sample = 0
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(testloader):
            inputs = inputs.to(device)
            # print(inputs.shape)
            classes = classes.to(device)
            outputs = resnet(inputs)
            ps = torch.exp(outputs)
            topk, topclass = ps.topk(1, dim=1)
            if sample < 6:
                if classes[0].item() != topclass.cpu().numpy()[0][0]:
                    print(
                        "True Label : ",
                        classes[0].item(),
                        "Prediction : ",
                        topclass.cpu().numpy()[0][0],
                        ", Score: ",
                        topk.cpu().numpy()[0][0],
                    )
                    ax = plt.subplot(2, 3, sample + 1)
                    plt.tight_layout()
                    ax.set_title("Sample #{}".format(sample))
                    ax.axis("off")
                    plt.imshow(inputs[0].cpu().permute(1, 2, 0))
                    sample += 1


if __name__ == "__main__":
    ### Dataset Creation
    CreateData()
    ### face Recognition
