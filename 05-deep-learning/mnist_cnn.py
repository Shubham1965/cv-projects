import numpy as np
import matplotlib.pyplot as plt

import torch 
from torch import nn, save, load
from torch.nn import functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor

# Download the training and testing datasets
train_data = datasets.MNIST(root="05-deep-learning/mnist-data",  train = True, download = True, transform = ToTensor())
test_data = datasets.MNIST(root='05-deep-learning/mnist-data', train=False, download=True, transform=ToTensor())

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare data loaders
batch_size = 32
number_of_workers = 0

train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=number_of_workers)
test_loader = DataLoader(test_data, batch_size=batch_size,  num_workers=number_of_workers)

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter._next_data()
images = images.numpy()

# visualize one image from the batch 
img = np.squeeze(images[0])
fig = plt.figure(figsize = (15,15)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x), horizontalalignment='center', verticalalignment='center', color='white' if img[x][y]<thresh else 'black')
        
# Define the MLP architecture
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    

if __name__ == '__main__':

    # Instantiate the model
    model = CNN()

    # Instantiate the loss function
    criterion = nn.CrossEntropyLoss()

    # Instantiate the optimizer
    optimizer = SGD(model.parameters(), lr=0.01)


    # Train the model
    num_epochs = 20
    model.train()

    for epoch in range(num_epochs):

        training_loss = 0.0

        for data, target in train_loader:

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()*data.size(0)

        print('Epoch {} \t Training loss: {:.6f}'.format(epoch+1, training_loss/len(train_loader.dataset)))

    
    # Test the model
    model.eval()

    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for data, target in test_loader:

        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()*data.size(0)

        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))

        for i in range(len(target)):

            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
            
    test_loss = test_loss/len(test_loader.dataset)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))


    
