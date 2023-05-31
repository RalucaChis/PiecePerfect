import os

import PIL
import torch
from PIL import Image
from torchvision import transforms
from siamese_network import SiameseNetwork


def find_top_5_matches(img_name, dir_path):
    model_path = "D:\\FACULTATE\\Master\\DISERTATIE\\PiecePerfect\\src\\edge_matching\\models\\inverted-second-img.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, device)
    img_path = os.path.join(dir_path, img_name)
    list_img = os.listdir(dir_path)
    list_img.remove(img_name)

    found_matches = []
    top_5 = {}
    img1 = Image.open(img_path).convert('L')
    w, h = 148, 148
    img1 = transforms.functional.crop(img1, 0, int(w / 2) + 10, h, int(w / 2) - 10)  # top, left, height, width
    img1 = transforms.Compose([
        transforms.ToTensor()
    ])(img1)
    img1_tensor = torch.Tensor.reshape(img1.to(device), (1, 1, 148, 64))
    with torch.no_grad():
        for img2_name in list_img:
            img2_path = os.path.join(dir_path, img2_name)
            img2 = PIL.ImageOps.invert(Image.open(img2_path).convert('L'))
            img2 = transforms.functional.crop(img2, 0, 0, h, int(w / 2) - 10)
            img2 = transforms.Compose([
                transforms.ToTensor()
            ])(img2)
            img2_tensor = torch.Tensor.reshape(img2.to(device), (1, 1, 148, 64))

            with torch.set_grad_enabled(False):
                outputs = model(img1_tensor, img2_tensor)
                top_5[float(outputs[0][1])] = img2_name
    sorted_keys = sorted(top_5.keys(), reverse=True)
    for key in sorted_keys[:5]:
        found_matches.append(top_5[key])
    return found_matches, top_5


if __name__ == "__main__":
    img_name = "vincent_1.JPG"
    masks_dir = '/\\static\\masks'
    matches, top_5 = find_top_5_matches(img_name, masks_dir)
    # print(top_5)
    print(matches)
