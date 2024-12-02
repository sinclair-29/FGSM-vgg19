import logging

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from vgg import vgg19_bn


path = './dataset'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s ",
    datefmt="%a %d %b %Y %H:%M:%S"
)


def denorm(batch, mean=[0.5], std=[0.5]):
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def generate_adv_sample(x, epsilon, x_grad):
    sign_x_grad = x_grad.sign()
    perturbed_x = x + epsilon * sign_x_grad
    perturbed_x = torch.clamp(perturbed_x, 0, 255)
    return perturbed_x


def test(model, loader, class_idx, num):
    count = 0
    epsilon_list, l2norm_list = [], []
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        if label.item() != class_idx:
            continue
        data.requires_grad = True
        output = model(data)
        predicted_label = output.max(dim=1, keepdim=True)[1]

        if predicted_label.item() != label.item():
            continue
        loss = F.nll_loss(output, label)
        model.zero_grad()
        loss.backward()
        x_grad = data.grad.data
        x_denorm = denorm(data)

        left, right = 0, 1.0
        DELTA = 0.01
        while right - left > DELTA:
            middle = (left + right) / 2.0
            perturbed_x = generate_adv_sample(x_denorm, middle, x_grad)
            hat_y = model(transforms.Normalize((0.5,), (0.5,))(perturbed_x))
            hat_label = hat_y.max(dim=1, keepdim=True)[1]
            if hat_label.item() == label.item():
                left = middle
            else:
                right = middle

        threshold = (left + right) / 2.0
        final_perturbed_x = generate_adv_sample(x_denorm, threshold, x_grad)
        epsilon_list.append(threshold)
        l2norm_list.append(torch.norm(final_perturbed_x - data, p=2).item())

        count += 1
        if count >= num:
            break

    for idx in range(len(epsilon_list)):
        logging.info(f'|class {class_idx} | {idx:2d}/{len(epsilon_list)} id'
                     f'| alpha {epsilon_list[idx]}'
                     f'| l2norm {l2norm_list[idx]}')


if __name__ == '__main__':
    model = vgg19_bn(num_classes=10).to(device)
    model.load_state_dict(torch.load("./model/model.pt", weights_only=True))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CIFAR10(root=path, train=False,
                                    download=True, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    for class_idx in range(10):
        test(model, test_dataloader, class_idx, num=10)
