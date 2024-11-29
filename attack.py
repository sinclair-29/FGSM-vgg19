

import torch

from vgg import vgg19_bn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_adv_sample():
    pass


def single_test():
    pass


def test(model, loader):
    pass


if __name__ == '__main__':
    model = vgg19_bn(num_classes=10).to(device)
    model.load_state_dict(torch.load("./model/model.pth"))