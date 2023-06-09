import os

import PIL
import torch
from PIL import Image
from torchvision import transforms
from siamese_network import SiameseNetwork


def find_top_5_matches(img_name, dir_path, selected_option):
    model_path = "D:\\FACULTATE\\Master\\DISERTATIE\\PiecePerfect\\src\\edge_matching\\models\\inverted-second-img.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, device)
    img_path = os.path.join(dir_path, img_name)
    list_img = os.listdir(dir_path)
    list_img.remove(img_name)

    found_matches = []
    match_probs = []
    top_5 = {}
    if selected_option == 'top':
        img1 = Image.open(img_path).convert('L').rotate(-90)
    elif selected_option == 'bottom':
        img1 = Image.open(img_path).convert('L').rotate(-270)
    elif selected_option == 'left':
        img1 = Image.open(img_path).convert('L').rotate(-180)
    else:
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
            if selected_option == 'top':
                img2 = PIL.ImageOps.invert(Image.open(img2_path).convert('L').rotate(-90))
            elif selected_option == 'bottom':
                img2 = PIL.ImageOps.invert(Image.open(img2_path).convert('L').rotate(-270))
            elif selected_option == 'left':
                img2 = PIL.ImageOps.invert(Image.open(img2_path).convert('L').rotate(-180))
            else:
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
    for key in sorted_keys:
        found_matches.append(top_5[key])
        match_probs.append(round(key * 100, 2))
    return found_matches, match_probs


if __name__ == "__main__":
    img_name = "vincent_13.JPG"
    masks_dir = 'D:\\FACULTATE\\Master\\DISERTATIE\\PiecePerfect\\static\\masks'
    matches, match_probs = find_top_5_matches(img_name, masks_dir, 'right-left')
    print(match_probs)
    print(matches)
