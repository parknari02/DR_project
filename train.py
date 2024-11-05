import sys
import os
import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import models
import shutil
from imblearn.over_sampling import RandomOverSampler


# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Python executable being used:", sys.executable)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())


# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 ])

# CSV 파일로부터 학습 데이터를 폴더 구조로 정리
def prepare_data(train_csv, train_images_dir, output_dir):
    labels_df = pd.read_csv(train_csv)
    os.makedirs(output_dir, exist_ok=True)
    for label in range(5):  # 0 ~ 4 라벨 디렉토리 생성
        os.makedirs(os.path.join(output_dir, str(label)), exist_ok=True)

    for _, row in labels_df.iterrows():
        file_name = row['image'] + '.jpeg'
        label = str(row['level'])
        src_path = os.path.join(train_images_dir, file_name)
        dst_path = os.path.join(output_dir, label, file_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            print(f"Copied: {src_path} to {dst_path}")
        else:
            print(f"File not found: {src_path}")

labels_df = pd.read_csv('trainLabels.csv')

file_names = labels_df['image'].values
labels = labels_df['level'].values

oversample = RandomOverSampler(sampling_strategy='auto')  # 자동으로 소수 클래스 오버샘플링
file_names_resampled, labels_resampled = oversample.fit_resample(file_names.reshape(-1, 1), labels)

resampled_labels_df = pd.DataFrame({
    'image': file_names_resampled.flatten(),
    'level': labels_resampled
})

# 학습 데이터 준비
prepare_data('trainLabels.csv', 'data/train', 'data/train_labeled_resampled')

# 데이터 로드 및 분할
full_dataset = ImageFolder(root='data/train_labeled_resampled', transform=transform)
sample_size = 8407
subset, _ = random_split(full_dataset, [sample_size, len(full_dataset) - sample_size])

train_size = int(0.8 * sample_size)
test_size = sample_size - train_size
train_dataset, test_dataset = random_split(subset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# EfficientNet-B0 모델 로드 및 수정
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)  # 5개 클래스 분류
model = model.to(device)

# 손실 함수 및 최적화기 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#학습 함수
def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train


        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_accuracy = 100 * correct_test / total_test

        # Epoch 결과 출력
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, '
                f'Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

    # 학습 완료 후 모델 저장
    torch.save(model.state_dict(), 'models/efficientnet_b0.pth')

# 모델 학습 실행
train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=10)
