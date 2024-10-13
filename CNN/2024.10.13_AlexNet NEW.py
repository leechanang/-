import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import time

class GrayscaleBMPDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.file_list = []
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            self.file_list.extend([(os.path.join(class_path, f), self.classes.index(class_name))
                                   for f in os.listdir(class_path) if f.endswith('.bmp')])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        image = Image.open(img_path).convert('RGB')  # RGB 이미지로 변환
        if self.transform:
            image = self.transform(image)
        return image, label

class HyperParameters:
    def __init__(self):
        self.num_epochs = 30
        self.batch_size = 8
        self.imgsz = 320  # AlexNet 입력 크기
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.weight_decay = 0.0005

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, results_folder):
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []
    best_val_auc = 0.0
    patience = 5
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_outputs = []

        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.detach().cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        all_labels_bin = label_binarize(all_labels, classes=range(3))
        train_auc = roc_auc_score(all_labels_bin, all_outputs, multi_class='ovr', average='macro')
        train_aucs.append(train_auc)

        print(f'Training Loss: {epoch_loss:.4f} AUC-ROC: {train_auc:.4f}')

        model.eval()
        val_loss = 0.0
        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        all_labels_bin = label_binarize(all_labels, classes=range(3))
        val_auc = roc_auc_score(all_labels_bin, all_outputs, multi_class='ovr', average='macro')
        val_aucs.append(val_auc)

        print(f'Validation Loss: {val_loss:.4f} AUC-ROC: {val_auc:.4f}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
            }, os.path.join(results_folder, 'best_model_checkpoint.pth'))
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        plt.figure(figsize=(10, 5))
        plt.title(f"Training and Validation AUC-ROC - Epoch {epoch + 1}")
        plt.plot(range(1, epoch + 2), train_aucs, label="Train")
        plt.plot(range(1, epoch + 2), val_aucs, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("AUC-ROC")
        plt.legend()
        plt.savefig(os.path.join(results_folder, f'auc_roc_curve_epoch_{epoch + 1}.png'))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.title(f"Training and Validation Loss - Epoch {epoch + 1}")
        plt.plot(range(1, epoch + 2), train_losses, label="Train")
        plt.plot(range(1, epoch + 2), val_losses, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(results_folder, f'loss_curve_epoch_{epoch + 1}.png'))
        plt.close()

    return best_val_auc

def evaluate_model(model, test_loader, device, classes, results_folder):
    model.eval()
    all_preds = []
    all_labels = []
    all_outputs = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    cm = confusion_matrix(all_labels, all_preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    all_labels_bin = label_binarize(all_labels, classes=range(3))
    auc_roc = roc_auc_score(all_labels_bin, all_outputs, multi_class='ovr', average='macro')

    with open(os.path.join(results_folder, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"AUC-ROC: {auc_roc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(results_folder, 'confusion_matrix_numbers.png'))
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (%)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(results_folder, 'confusion_matrix_percent.png'))
    plt.close()

    num_examples = min(5, len(test_loader.dataset))
    fig, axes = plt.subplots(1, num_examples, figsize=(20, 4))
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if i >= num_examples:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            img = inputs[0].cpu().numpy().transpose(1, 2, 0)
            axes[i].imshow(img)
            axes[i].set_title(f"True: {classes[labels[0]]}\nPred: {classes[predicted[0]]}")
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'classification_examples.png'))
    plt.close()

    return accuracy, auc_roc

if __name__ == "__main__":
    start_time = time.time()

    image_folder = "D:/dream_dataset/split3"
    results_folder = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(results_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hp = HyperParameters()

    transform = transforms.Compose([
        transforms.Resize((hp.imgsz, hp.imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = GrayscaleBMPDataset(os.path.join(image_folder, 'train'), transform=transform)
    val_dataset = GrayscaleBMPDataset(os.path.join(image_folder, 'val'), transform=transform)
    test_dataset = GrayscaleBMPDataset(os.path.join(image_folder, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hp.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=hp.batch_size)

    print("Dataset sizes:")
    print(f"Train: {len(train_dataset)}")
    print(f"Validation: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")

    model = models.alexnet(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=hp.learning_rate, momentum=hp.momentum, weight_decay=hp.weight_decay)

    print("Starting training...")
    best_val_auc = train_model(model, train_loader, val_loader, criterion, optimizer, hp.num_epochs, device, results_folder)

    checkpoint = torch.load(os.path.join(results_folder, 'best_model_checkpoint.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best validation AUC-ROC: {best_val_auc:.4f}")

    print("Evaluating best model...")
    test_accuracy, test_auc_roc = evaluate_model(model, test_loader, device, ['class1', 'class2', 'class3'], results_folder)

    torch.save({
        'model_state_dict': model.state_dict(),
        'best_val_auc_roc': best_val_auc,
        'test_accuracy': test_accuracy,
        'test_auc_roc': test_auc_roc
    }, os.path.join(results_folder, 'final_model.pth'))

    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    with open(os.path.join(results_folder, 'total_execution_time.txt'), 'w') as f:
        f.write(f"Total execution time: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")

    print(f"Total execution time: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")
    print(f"All results have been saved in {results_folder}")