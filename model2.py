import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

# 데이터 로딩 및 전처리
def load_audio(file_path):
    audio, _ = librosa.load(file_path, sr=22050)  # 22.05 kHz로 오디오 로드
    return audio

def extract_mel_spectrogram(audio):
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=22050)
    log_mel_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return log_mel_spectrogram

# 모델 정의
class GolfBallClassifier(nn.Module):
    def __init__(self):
        super(GolfBallClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 54 * 54, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# 하이퍼파라미터 설정
input_height = 128
input_width = 128
learning_rate = 0.001
batch_size = 16
num_epochs = 10

# 데이터 로드 및 전처리
normal_audio_paths = ['path/to/normal_1.wav', 'path/to/normal_2.wav', ...]  # 정상 골프공 충격음 파일 경로들
broken_audio_paths = ['path/to/broken_1.wav', 'path/to/broken_2.wav', ...]  # 깨진 골프공 충격음 파일 경로들

# 데이터셋 분할
all_audio_paths = normal_audio_paths + broken_audio_paths
all_labels = [0] * len(normal_audio_paths) + [1] * len(broken_audio_paths)
train_audio_paths, test_audio_paths, train_labels, test_labels = train_test_split(all_audio_paths, all_labels, test_size=0.2, random_state=42)

# 데이터셋 클래스
class GolfBallDataset(data.Dataset):
    def __init__(self, audio_paths, labels):
        self.audio_paths = audio_paths
        self.labels = labels

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        label = self.labels[index]

        audio = load_audio(audio_path)
        mel_spectrogram = extract_mel_spectrogram(audio)

        # Mel spectrogram을 3차원으로 변환하여 PyTorch 텐서로 반환
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
        mel_spectrogram = torch.from_numpy(mel_spectrogram).float()

        return mel_spectrogram, label

    def __len__(self):
        return len(self.audio_paths)

# 훈련 및 평가 함수
def train(model, dataloader, criterion, optimizer):
    model.train()
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# 데이터로더 생성
train_dataset = GolfBallDataset(train_audio_paths, train_labels)
test_dataset = GolfBallDataset(test_audio_paths, test_labels)

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 및 손실 함수, 옵티마이저 생성
model = GolfBallClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# CUDA 사용 가능 여부 확인 후 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 훈련 및 평가
for epoch in range(num_epochs):
    train(model, train_dataloader, criterion, optimizer)
    train_accuracy = evaluate(model, train_dataloader)
    test_accuracy = evaluate(model, test_dataloader)
    print(f"Epoch {epoch+1}: Train Accuracy = {train_accuracy:.2f}%, Test Accuracy = {test_accuracy:.2f}%")