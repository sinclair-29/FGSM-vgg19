import logging
import matplotlib.pyplot as plt
from cycler import cycler

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from vgg import vgg19_bn


path = './dataset'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s ",
    datefmt="%a %d %b %Y %H:%M:%S"
)


def denorm(batch, mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]):
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def normalize(batch, mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]):
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)
    return (batch - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)



def generate_adv_sample(x, epsilon, x_grad):
    sign_x_grad = x_grad.sign()
    perturbed_x = x + epsilon * sign_x_grad
    perturbed_x = torch.clamp(perturbed_x, 0, 1)
    return perturbed_x


def BIM_test(model, loader, class_idx, num):
    count = 0
    iter_count_list, prob_list = [], []
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        if label.item() != class_idx:
            continue

        ITER_NUM = 100
        EPSILON = 1.0 / 255

        for iter_count in range(ITER_NUM):
            data.requires_grad = True
            output = model(data)
            predicted_label = output.max(dim=1, keepdim=True)[1]
            if iter_count == 0 and predicted_label.item() != label.item():
                continue

            probabilities = F.softmax(output, dim=1)
            prob = probabilities[0][predicted_label].item()
            output = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output, label)

            model.zero_grad()
            loss.backward()
            x_grad = data.grad.data
            x_denorm = denorm(data)
            data = generate_adv_sample(x_denorm, EPSILON, x_grad)
            data.detach_()
            
            
            hat_y = model(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))(data))
            hat_label = hat_y.max(dim=1, keepdim=True)[1]
            if hat_label.item() == label.item():
                data = normalize(data)
                continue
            else:
                prob_list.append(prob)
                iter_count_list.append(iter_count + 1)
                count += 1
                break

        if count >= num:
            break

    for idx in range(len(prob_list)):
        logging.info(f'| class {class_idx} | {idx + 1:2d}/{len(prob_list)} id'
                     f'| iter num {iter_count_list[idx]:2d}'
                     f'| probability {prob_list[idx]:.6f}')
    return iter_count_list, prob_list




def test(model, loader, class_idx, num):
    count = 0
    epsilon_list, l2norm_list, prob_list = [], [], []
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        if label.item() != class_idx:
            continue
        data.requires_grad = True
        output = model(data)
        predicted_label = output.max(dim=1, keepdim=True)[1]
        if predicted_label.item() != label.item():
            continue

        probabilities = F.softmax(output, dim=1)
        prob = probabilities[0][predicted_label].item()
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, label)
        #loss = nn.CrossEntropyLoss(output, label)
        model.zero_grad()
        loss.backward()
        x_grad = data.grad.data
        x_denorm = denorm(data)

        left, right = 0, 1.0
        DELTA = 0.01
        while right - left > DELTA:
            middle = (left + right) / 2.0
            perturbed_x = generate_adv_sample(x_denorm, middle, x_grad)
            hat_y = model(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))(perturbed_x))
            hat_label = hat_y.max(dim=1, keepdim=True)[1]
            if hat_label.item() == label.item():
                left = middle
            else:
                right = middle

        threshold = (left + right) / 2.0
        if threshold > 0.35:
            continue
        prob_list.append(prob)
        final_perturbed_x = generate_adv_sample(x_denorm, threshold, x_grad)
        epsilon_list.append(threshold)
        l2norm_list.append(torch.norm(final_perturbed_x - x_denorm, p=2).item())

        count += 1
        if count >= num:
            break

    for idx in range(len(epsilon_list)):
        logging.info(f'| class {class_idx} | {idx+1:2d}/{len(epsilon_list)} id'
                     f'| alpha {epsilon_list[idx]:.6f}'
                     f'| l2norm {l2norm_list[idx]:.6f}'
                     f'| probability {prob_list[idx]:.6f}')

    return epsilon_list, prob_list


if __name__ == '__main__':
    model = vgg19_bn(num_classes=10).to(device)
    model.load_state_dict(torch.load("./model/model.pt", weights_only=True))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    test_dataset = datasets.CIFAR10(root=path, train=False,
                                    download=True, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    #pltx_list, plty_list = [], []
    plt.xlabel("probability")
    plt.ylabel("iter num")
    for class_idx in range(10):
        epsilon_list, prob_list = BIM_test(model, test_dataloader, class_idx, num=10)
        plt.scatter(prob_list, epsilon_list)
        plt.savefig(f'./plot/scatter_plot{class_idx}.png')  
