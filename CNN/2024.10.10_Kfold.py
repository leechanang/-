import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# 하이퍼파라미터 설정
IMAGE_SIZE = 332
LEARNING_RATE = 0.001
BATCH_SIZE = 8
EPOCHS = 30
NUM_FOLDS = 5
EARLY_STOP_PATIENCE = 10
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def histogram_equalization(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    max_val = np.max(equalized_image)
    if max_val > 150:
        equalized_image = (equalized_image / max_val * 150).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB))


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = histogram_equalization(image)
        label = self.labels[idx]

        if self.augment:
            image = self.augment_image(image)

        if self.transform:
            image = self.transform(image)

        return image, label

    def augment_image(self, image):
        augment_transform = transforms.Compose([
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0))
        ])
        return augment_transform(image)


def load_data(data_dir):
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(data_dir))

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, image_name))
            labels.append(class_idx)

    return image_paths, labels, class_names


def balance_dataset(image_paths, labels):
    class_counts = np.bincount(labels)
    max_count = np.max(class_counts)
    balanced_image_paths = []
    balanced_labels = []

    for class_idx in range(len(class_counts)):
        class_image_paths = [path for path, label in zip(image_paths, labels) if label == class_idx]
        class_image_paths = class_image_paths * (max_count // len(class_image_paths))
        class_image_paths += np.random.choice(class_image_paths, max_count - len(class_image_paths),
                                              replace=False).tolist()

        balanced_image_paths.extend(class_image_paths)
        balanced_labels.extend([class_idx] * max_count)

    return balanced_image_paths, balanced_labels


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, early_stop_patience):
    best_val_accuracy = 0
    patience_counter = 0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return best_val_accuracy, train_losses, train_accuracies, val_losses, val_accuracies


def plot_learning_curves(train_losses, train_accuracies, val_losses, val_accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Count)')
    plt.tight_layout()
    plt.savefig('confusion_matrix_count.png')
    plt.close()

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Ratio)')
    plt.tight_layout()
    plt.savefig('confusion_matrix_ratio.png')
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "D:/dream_dataset/images"
    image_paths, labels, class_names = load_data(data_dir)

    balanced_image_paths, balanced_labels = balance_dataset(image_paths, labels)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    start_time = time.time()

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(balanced_image_paths, balanced_labels), 1):
        print(f"Fold {fold}")

        train_idx = train_val_idx[:int(len(train_val_idx) * (TRAIN_RATIO / (TRAIN_RATIO + VAL_RATIO)))]
        val_idx = train_val_idx[int(len(train_val_idx) * (TRAIN_RATIO / (TRAIN_RATIO + VAL_RATIO))):]

        train_dataset = CustomDataset([balanced_image_paths[i] for i in train_idx],
                                      [balanced_labels[i] for i in train_idx], transform, augment=True)
        val_dataset = CustomDataset([balanced_image_paths[i] for i in val_idx], [balanced_labels[i] for i in val_idx],
                                    transform)
        test_dataset = CustomDataset([balanced_image_paths[i] for i in test_idx],
                                     [balanced_labels[i] for i in test_idx], transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val_accuracy, train_losses, train_accuracies, val_losses, val_accuracies = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, EPOCHS, EARLY_STOP_PATIENCE
        )

        plot_learning_curves(train_losses, train_accuracies, val_losses, val_accuracies)

        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(test_labels, test_preds)
        precision = precision_score(test_labels, test_preds, average='weighted')
        recall = recall_score(test_labels, test_preds, average='weighted')
        f1 = f1_score(test_labels, test_preds, average='weighted')

        plot_confusion_matrix(test_labels, test_preds, class_names)

        fold_results.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'val_accuracy': best_val_accuracy
        })

    end_time = time.time()
    total_time = end_time - start_time

    print("\nFinal Results:")
    for result in fold_results:
        print(f"Fold {result['fold']}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall: {result['recall']:.4f}")
        print(f"  F1 Score: {result['f1']:.4f}")
        print(f"  Validation Accuracy: {result['val_accuracy']:.4f}")

    print(f"\nTotal training time: {total_time:.2f} seconds")

    with open('results.txt', 'w') as f:
        for result in fold_results:
            f.write(f"Fold {result['fold']}:\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"  Precision: {result['precision']:.4f}\n")
            f.write(f"  Recall: {result['recall']:.4f}\n")
            f.write(f"  F1 Score: {result['f1']:.4f}\n")
            f.write(f"  Validation Accuracy: {result['val_accuracy']:.4f}\n")
        f.write(f"\nTotal training time: {total_time:.2f} seconds\n")


if __name__ == "__main__":
    main()