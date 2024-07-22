import torch
import random
import time
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader

# import VisionTransformer model from torchvision models
from torchvision.models.vision_transformer import VisionTransformer

# MixUp class to implement MixUp data augmentation technique
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
        mixed_xT = lmbda * xT + (1 - lmbda) * xT[index, :]
        yT_a, yT_b = yT, yT[index]
        
        return mixed_xT, yT_a, yT_b, lmbda
    
# criterion for calculating loss with MixUp
def mixup_criterion(criterion, pred, yT_a, yT_b, lmbda):
    return lmbda * criterion(pred, yT_a) + (1 - lmbda) * criterion(pred, yT_b)

# function to calculate F1 score for multi-class classification
def calculate_f1_score(targets, predictions, num_classes):
    true_positives = np.zeros(num_classes)
    false_positives = np.zeros(num_classes)
    false_negatives = np.zeros(num_classes)
    
    # compute true positives, false positives, and false negatives for each class
    for i in range(num_classes):
        true_positives[i] = np.sum((predictions == i) & (targets == i))
        false_positives[i] = np.sum((predictions == i) & (targets != i))
        false_negatives[i] = np.sum((predictions != i) & (targets == i))
    
    # calculate precision, recall, and F1 score for each class
    precision = true_positives / (true_positives + false_positives + 1e-7)
    recall = true_positives / (true_positives + false_negatives + 1e-7)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    # compute macro F1 score as the mean of the class-wise F1 scores
    macro_f1 = np.mean(f1_scores)
    return macro_f1



if __name__ == '__main__':
    
    # data augmentation and normalization for CIFAR-10
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 20
    
    # load both the training and test datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # combine the trainset and testset
    combined_set = torch.utils.data.ConcatDataset([trainset, testset])

    # perform the split into development set (80%) and holdout test set (20%)
    total_size = len(combined_set)
    dev_size = int(0.8 * total_size)
    holdout_test_size = total_size - dev_size
    dev_set, holdout_test_set = torch.utils.data.random_split(combined_set, [dev_size, holdout_test_size])

    # further split the development set into training (90%) and validation set (10%)
    train_size = int(0.9 * len(dev_set))
    validation_size = len(dev_set) - train_size
    train_set, validation_set = torch.utils.data.random_split(dev_set, [train_size, validation_size])

    # define DataLoaders for the new splits
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    validationloader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=2)
    holdout_testloader = DataLoader(holdout_test_set, batch_size=batch_size, shuffle=False, num_workers=2)


    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    save_image_flag = True

    # initialize the Vision Transformer model
    net = VisionTransformer(image_size=32, patch_size=8, num_layers=3, num_heads=3,hidden_dim=192,mlp_dim=768, num_classes=10) # changed

    # set loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # initialize MixUp with a chosen method
    sampling_method = 2
    mixup = MixUp(method=sampling_method, alpha=0.2)

    # training loop
    best_f1 = 0.0

    for epoch in range(20): 
        
        running_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        epoch_start_time = time.time()
        
        all_train_targets = []
        all_train_predictions = []
    
        for i, data in enumerate(trainloader, 0):
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # apply mixup augumentation
            inputs, targets_a, targets_b, lmbda = mixup.mixup(inputs, labels)
            
            # forward + backward + optimize
            outputs = net(inputs)
            
            # loss = criterion(outputs, labels)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lmbda)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total_samples += inputs.size(0)
        
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            
            all_train_targets.extend(labels.numpy())
            all_train_predictions.extend(predicted.numpy())
        
        if epoch == 19:
            save_path = f'task3/sampling_method_{sampling_method}_model.pt'  # save models at specific epochs
            torch.save(net.state_dict(), save_path)
        
        train_f1_score = calculate_f1_score(np.array(all_train_targets), np.array(all_train_predictions), num_classes=len(classes))
        avg_train_loss = running_loss / total_samples
        train_accuracy = correct_predictions / total_samples * 100
        
        # validation performance calculation as before, then calculate validation F1 and loss
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # validation loop
        validation_loss = 0.0
        total_val_samples = 0
        correct_val_predictions = 0
        all_val_labels = []
        all_val_predictions = []

        with torch.no_grad():
            for inputs, labels in validationloader:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item() * inputs.size(0)
                total_val_samples += inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                correct_val_predictions += (predicted == labels).sum().item()
                
                all_val_labels.extend(labels.detach().numpy())
                all_val_predictions.extend(predicted.detach().numpy())
        
        avg_val_loss = validation_loss / total_val_samples
        val_accuracy = correct_val_predictions / total_val_samples * 100
        val_f1_score = calculate_f1_score(np.array(all_val_labels), np.array(all_val_predictions), num_classes=len(classes))

        print(f'Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f}, '
            f'Training Accuracy: {train_accuracy:.2f}%, '
            f'Training F1 Score: {train_f1_score:.4f}, '
            f'Validation Loss: {avg_val_loss:.4f}, '
            f'Validation Accuracy: {val_accuracy:.2f}%, '
            f'Validation F1 Score: {val_f1_score:.4f}, '
            f'Speed: {epoch_duration:.2f} seconds')

    print('Training done.')

        
    model_path = f'task3/sampling_method_{sampling_method}_model.pt'
    net.load_state_dict(torch.load(model_path))
    
    correct = 0
    total = 0
    
    # holdout test set evaluation
    holdout_loss = 0.0
    holdout_total = 0
    holdout_correct = 0
    all_holdout_targets = []
    all_holdout_predictions = []
    
    with torch.no_grad():
        for data in holdout_testloader:
            images, labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)
            holdout_loss += loss.item() * inputs.size(0)
            holdout_total += inputs.size(0)
    
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            holdout_correct += (predicted == labels).sum().item()
    
            all_holdout_targets.extend(labels.numpy())
            all_holdout_predictions.extend(predicted.numpy())
    
    accuracy = 100 * correct / total

    holdout_f1_score = calculate_f1_score(np.array(all_holdout_targets), np.array(all_holdout_predictions), num_classes=len(classes))
    avg_holdout_loss = holdout_loss / holdout_total
    holdout_accuracy = holdout_correct / holdout_total * 100

    print(f'Holdout Test Set: Loss: {avg_holdout_loss:.4f}, Accuracy: {holdout_accuracy:.2f}%, F1 Score: {holdout_f1_score:.4f}')

