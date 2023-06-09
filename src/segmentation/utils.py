import cv2
import torch
import torchvision
from PIL.Image import Image
from PIL import Image
from torchvision.transforms import transforms
from src.segmentation.u_net_network import U_Net, encoder_block, decoder_block, conv_block


def generate_mask(img_path, save_path):
    model_path = "D:\\FACULTATE\\Master\\DISERTATIE\\PiecePerfect\\src\\segmentation\\models\\u-net-lr-0-003.pt"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(model_path, device)

    img = Image.open(img_path).convert('RGB')
    img = transforms.Compose([
        transforms.ToTensor()
    ])(img)
    img_tensor = torch.Tensor.reshape(img.to(device), (1, 3, 256, 256))

    with torch.set_grad_enabled(False):
        output = model(img_tensor)
        for sublist in range(len(output)):
            for subsublist in range(len(output[sublist])):
                for sss in range(len(output[sublist][subsublist])):
                    for item in range(len(output[sublist][subsublist][sss])):
                        if float(output[sublist][subsublist][sss][item]) > 0.99:
                            output[sublist][subsublist][sss][item] = 1
                        else:
                            output[sublist][subsublist][sss][item] = 0
        mask = torchvision.transforms.ToPILImage()(output[0])
        mask.save(save_path)


def crop_around_piece(img_path):
    img = cv2.imread(img_path, 2)
    cropped = img.copy()
    top_x, top_y = 0, 0
    bottom_x, bottom_y = 0, 0
    left_x, left_y = 0, 0
    right_x, right_y = 0, 0
    for i in range(len(img)):
        for j in range(len(img[i])):
            if cropped[i][j] == 255 and (cropped[i + 1][j] == 255 or cropped[i][j + 1] == 255):
                top_x, top_y = i, j
                break
        if top_x != 0 and top_y != 0:
            break

    for i in range(len(img) - 1, 0, -1):
        for j in range(len(img[i]) - 1, 0, -1):
            if cropped[i][j] == 255 and (cropped[i - 1][j] == 255 or cropped[i][j - 1] == 255):
                bottom_x, bottom_y = i, j
                break
        if bottom_x != 0 and bottom_y != 0:
            break

    for i in range(len(img)):
        for j in range(len(img[i])):
            if cropped[j][i] == 255 and (cropped[j + 1][i] == 255 or cropped[j][i + 1] == 255):
                left_x, left_y = j, i
                break
        if left_x != 0 and left_y != 0:
            break

    for i in range(len(img) - 1, 0, -1):
        for j in range(len(img[i])):
            if cropped[j][i] == 255 and (cropped[j - 1][i] == 255 or cropped[j][i - 1] == 255):
                right_x, right_y = j, i
                break
        if right_x != 0 and right_y != 0:
            break

    cropped = cropped[top_x - 5:bottom_x + 5, left_y - 5:right_y + 5]
    # print('Cropped Dimensions : ', cropped.shape)
    cv2.imwrite(img_path, cropped)


def padding_image(image, padding_values, padding_color):
    top, bottom, left, right = padding_values
    width, height = image.size
    width2, height2 = width + left + right, height + top + bottom

    padded_image = Image.new(mode="RGB", size=(width2, height2), color=padding_color)
    padded_image.paste(image, box=(left, top))
    return padded_image


def add_padding_reshape_to_148(path_img):
    new_dim = 148
    img = Image.open(path_img)

    diff_w1 = new_dim - img.size[0]
    diff_h1 = new_dim - img.size[1]
    if diff_w1 % 2 == 1:
        diff_w1 = (new_dim - img.size[0]) // 2
        diff_w2 = (new_dim - img.size[0]) // 2 + 1
    else:
        diff_w1 = (new_dim - img.size[0]) // 2
        diff_w2 = (new_dim - img.size[0]) // 2
    if diff_h1 % 2 == 1:
        diff_h1 = (new_dim - img.size[1]) // 2
        diff_h2 = (new_dim - img.size[1]) // 2 + 1
    else:
        diff_h1 = (new_dim - img.size[1]) // 2
        diff_h2 = (new_dim - img.size[1]) // 2

    padded = padding_image(img, (diff_h1, diff_h2, diff_w1, diff_w2), (0, 0, 0))
    padded.save(path_img)


if __name__ == "__main__":
    img_path = "D:\\FACULTATE\\Master\\DISERTATIE\\PiecePerfect\\static\\puzzle_pieces\\vincent_13.JPG"
    save_path = "D:\\FACULTATE\\Master\\DISERTATIE\\PiecePerfect\\static\\masks\\vincent_13.JPG"
    generate_mask(img_path, save_path)
    crop_around_piece(save_path)
    add_padding_reshape_to_148(save_path)

    # mask.save(save_path)
    # mask.show()
