import torch
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader

# from network_pt import Net
from torchvision.models.vision_transformer import VisionTransformer

# set the seed for reproducibility
seed_value = 42 
random.seed(seed_value)
np.random.seed(seed_value)

# function to save mixed images to a specified path
def save_mixup_images(mixed_images, path="task2/mixup.png", nrow=4, normalize=True):

    images_to_save = mixed_images[:nrow*nrow]
    grid = vutils.make_grid(images_to_save, nrow=nrow, normalize=normalize)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

# MixUp augmentation class
class MixUp:
    
    # initialization with method and alpha parameters for MixUp
    def __init__(self, method, alpha):
        
        self.method = method
        self.alpha = alpha
    
    # mixup function to blend images and labels
    def mixup(self, xT, yT):
        
        assert self.method in [1, 2], "Method must be 1 or 2"
        
        alpha_range = [0, 1]
        
        # decide the lambda value based on the chosen method
        if self.method == 1:
            lmbda = np.random.beta(self.alpha, self.alpha)
        elif self.method == 2:
            lmbda = np.random.uniform(alpha_range[0], alpha_range[1])
        
        # shuffle and mix inputs based on lambda
        batch_size = xT.size()[0]
        index = torch.randperm(batch_size)
        mixed_xT = lmbda * xT + (1 - lmbda) * xT[index, :] # look at pytorch doc implementation for the next ones 
        yT_a, yT_b = yT, yT[index]
        
        return mixed_xT, yT_a, yT_b, lmbda

# criterion for calculating loss with MixUp
def mixup_criterion(criterion, pred, yT_a, yT_b, lmbda):
    return lmbda * criterion(pred, yT_a) + (1 - lmbda) * criterion(pred, yT_b)

if __name__ == '__main__':
    
        
    # data augmentation and normalization for CIFAR-10
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 20
    
    # load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    save_image_flag = True

    # initialize the Vision Transformer model
    net = VisionTransformer(image_size=32, patch_size=8, num_layers=3, num_heads=3,hidden_dim=192,mlp_dim=768, num_classes=10) # changed

    # set loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # initialize MixUp with a chosen method
    sampling_method = 1
    mixup = MixUp(method=sampling_method, alpha=0.2)

    # training loop
    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # apply mixup augumentation
            inputs, targets_a, targets_b, lmbda = mixup.mixup(inputs, labels)
            if save_image_flag:
                save_mixup_images(inputs, path=f'task2/mixup_sampling_method_{sampling_method}.png', nrow=4, normalize=True)
                save_image_flag = False
            
            # forward + backward + optimize
            outputs = net(inputs)
            
            # loss = criterion(outputs, labels)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lmbda)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                
        if epoch in [0, 4, 9, 14, 19]:
            save_path = f'task2/sampling_method_{sampling_method}_model_epoch_{epoch+1}.pt'  # save models at specific epochs
            torch.save(net.state_dict(), save_path)

    print('Training done.')

    # testing loop
    epochs_to_test = [1, 5, 10, 15, 20]
    batch_size = 36
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    for epoch in epochs_to_test:
        
        model_path = f'task2/sampling_method_{sampling_method}_model_epoch_{epoch}.pt'
        net.load_state_dict(torch.load(model_path))
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Accuracy of the model at epoch {epoch}: {accuracy:.2f}%')
    
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    # save the images as a single row
    save_mixup_images(images, f'task2/result_sampling_method_{sampling_method}.png', nrow=36, normalize=True)

    # print the labels as required
    print('Ground-truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(36)))
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(36)))
