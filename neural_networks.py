import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 데이터 준비
INPUT_PATH = './open/'

train = pd.read_csv(f'{INPUT_PATH}train.csv', parse_dates=['사고일시'])
test = pd.read_csv(f'{INPUT_PATH}test.csv', parse_dates=['사고일시'])
ss = pd.read_csv(f'{INPUT_PATH}sample_submission.csv')


y_train = train['ECLO']
cols = test.columns
X_train = train[cols]
X_test = test


def feat_eng(df):
    df['월'] = df['사고일시'].dt.month
    df['일'] = df['사고일시'].dt.day
    df['시'] = df['사고일시'].dt.hour

    subs = ['ID', '사고일시', '기상상태', '시군구']
    df = df.drop(subs, axis=1)

    df = pd.get_dummies(df)

    return df


X_train_eng = feat_eng(X_train)
X_test_eng = feat_eng(X_test)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_eng, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 데이터를 PyTorch의 Tensor 형태로 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.long)

# 데이터로더 생성
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 모델 정의
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, y_train.max().item() + 1),
)

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 손실함수와 최적화 함수 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 학습
model.train()
for epoch in range(100):
    for i, (X_batch, y_batch) in enumerate(train_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

# 검증
model.eval()
with torch.no_grad():
    X_val, y_val = X_val.to(device), y_val.to(device)
    y_pred = model(X_val)
    _, predicted = torch.max(y_pred, 1)
    accuracy = (predicted == y_val).sum().item() / y_val.size(0)

print(f'Accuracy: {accuracy}')


# 테스트 데이터 준비
X_test_eng = scaler.transform(X_test_eng)
X_test_eng = torch.tensor(X_test_eng, dtype=torch.float32)

# 테스트 데이터 예측
model.eval()
with torch.no_grad():
    X_test_eng = X_test_eng.to(device)
    y_pred = model(X_test_eng)
    _, predicted = torch.max(y_pred, 1)

# 예측 결과를 CPU의 NumPy 배열로 변환
predicted = predicted.cpu().numpy()

# 예측 결과를 제출 파일에 저장
ss['ECLO'] = predicted

OUTPUT_PATH = './out'
ss.to_csv(f'{OUTPUT_PATH}submission.csv', index=False)
