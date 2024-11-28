import logging

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from vgg import vgg19_bn


path = './dataset'
BATCH_SIZE = 64
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s ",
    datefmt="%a %d %b %Y %H:%M %S"
)


def train(model, data_loader, criterion, optimizer, log_interval = 100):
    model.train()
    total_loss, total_acc, total_count = 0, 0, 0
    for idx, (data, label) in enumerate(data_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        predicted_label = output.data.max(dim=1, keepdim=True)[1]
        total_acc += predicted_label.eq(label.data.view_as(predicted_label)).cpu().sum()
        total_count += data.shape[0]
        total_loss += loss.item()

        if idx > 0 and idx % log_interval == 0:
            logging.info(f'| epoch {epoch} | {idx:2d} / {len(data_loader)} batches'
                         f'| accuracy {total_acc / total_count:.4f}'
                         f'| loss {total_loss / total_count:.4f}')
            total_acc, total_count, total_loss = 0, 0 ,0


def eval(model, data_loader):
    model.eval()
    num_correct, total_loss = 0, 0
    with torch.no_grad():
        for idx, (data, label) in enumerate(data_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            total_loss += loss
            predicted_label = output.data.max(dim=1, keepdim=True)[1]
            num_correct += predicted_label.eq(label.data.view_as(predicted_label)).cpu().sum()
            current_batch_size = data.size(0)

    accuracy = 100. * num_correct / len(data_loader.dataset)
    logging.info(f'Accuracy on test dataset: {accuracy}%')
    eval_loss = loss / len(data_loader)
    logging.info(f'loss of test set: {eval_loss}')


if __name__ == '__main__':
    model = vgg19_bn(num_classes=10).to(device)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root=path, train=True,
                                     download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=path, train=False,
                                    download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    num_epoch = 100
    for epoch in range(num_epoch):
        train(model, train_dataloader, criterion, optimizer)
        eval(model, test_dataloader)
    eval(model, test_dataloader)
