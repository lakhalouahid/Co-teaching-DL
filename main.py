import torch
import math
import torch.nn.functional as F
from torchvision.transforms import Normalize, Compose, ToTensor
from torch.utils.data import DataLoader
from model import CNN
from loss import loss_coteaching
from dataset.cifar import CIFAR10

# Constants
batch_size = 256
input_channel = 3
num_classes = 10
learning_rate = 5e-4
model_str = 'CIFAR10'
n_epoch = 40
epoch_decay_start = 4
noise_type = 'symmetric'
noise_rate = 0.2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]

# define drop rate schedule
forget_rate = noise_rate


alpha_plan = torch.ones(n_epoch) * learning_rate

for i in range(epoch_decay_start, n_epoch):
    alpha_plan[i] = float(n_epoch - i) / (n_epoch - epoch_decay_start) * learning_rate

def adjust_learning_rate(optimizer, epoch_idx):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch_idx] / (float(math.sqrt(epoch_idx+1)))



@torch.no_grad()
def accuracy(model, data_loader):
    """Computes the accuracy of a model"""
    model.eval()
    total, correct = 0, 0
    for (images, labels, _) in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        ypred = model(images)
        output = F.softmax(ypred, dim=1)
        correct += torch.sum(torch.argmax(output, dim=1) == labels)
        total += labels.shape[0]
    return float(correct / total)

def accuracy_batch(ypred, labels):
    """Computes the accuracy of a model"""
    output = F.softmax(ypred, dim=1)
    correct = torch.sum(torch.argmax(output, dim=1) == labels)
    return float(correct / labels.shape[0])

def train(train_loader, test_loader, epoch_idx, model1, optimizer1, model2, optimizer2):
    """Train the two models using Co-teaching method"""
    print(f'epoch {epoch_idx+1}/{n_epoch}')
    remember_rate = 1 - max(forget_rate, forget_rate * (epoch_idx/epoch_decay_start))
    for batch_idx, (images, labels, _) in enumerate(train_loader):
        num_remember = int(remember_rate * labels.shape[0])
        images = images.to(device)
        labels = labels.to(device)
        logits1 = model1(images)
        logits2 = model2(images)
        loss_1, loss_2 = loss_coteaching(logits1, logits2, labels, num_remember)
        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
    accuracy1, accuracy2 = evaluate(test_loader, model1, model2)
    print(f"accuracy of model 1 on test set: {accuracy1*100}\naccuracy of model 2 on test set: {accuracy2*100}")

@torch.no_grad()
def evaluate(test_loader, model1, model2):
    """Evaluate model1, model2"""
    model1.eval()
    total, correct1, correct2 = 0, 0, 0
    for (images, labels, _) in test_loader:
        total += labels.shape[0]
        images = images.to(device)
        labels = labels.to(device)
        ypred1 = model1(images)
        probs1 = F.softmax(ypred1, dim=1)
        correct1 += torch.sum(torch.argmax(probs1, dim=1) == labels)
        ypred2 = model2(images)
        probs2 = F.softmax(ypred2, dim=1)
        correct2 += torch.sum(torch.argmax(probs2, dim=1) == labels)
    return float(correct1 / total), float(correct2 / total)

# Construting TensorDatasets
print('Construting datasets ...')
train_dataset = CIFAR10('data', train=True, download=True, noise_type=noise_type, noise_rate=noise_rate, transform=Compose([
    ToTensor(),
    Normalize(mean=mean, std=std)]))
test_dataset = CIFAR10('data', train=False, download=True, noise_type=noise_type, noise_rate=noise_rate, transform=Compose([
    ToTensor(),
    Normalize(mean=mean, std=std)]))

# Construting DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)

cnn1 = CNN(input_channel=input_channel, n_outputs=num_classes).to(device)
cnn2 = CNN(input_channel=input_channel, n_outputs=num_classes).to(device)

optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)
optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=learning_rate)

# Train modelsn_epoch
n_epoch = min(n_epoch, int(epoch_decay_start/forget_rate))
for epoch_idx in range(n_epoch):
    cnn1.train()
    cnn2.train()
    adjust_learning_rate(optimizer1, epoch_idx)
    adjust_learning_rate(optimizer2, epoch_idx)
    train(train_loader, test_loader, epoch_idx, cnn1, optimizer1, cnn2, optimizer2)
