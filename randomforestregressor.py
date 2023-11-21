import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error


INPUT_PATH = './open/'

train = pd.read_csv(f'{INPUT_PATH}train.csv')
test = pd.read_csv(f'{INPUT_PATH}test.csv')
ss = pd.read_csv(f'{INPUT_PATH}sample_submission.csv')


def making_val_table(df):
    '''입력받은 데이터셋의 유효성 검증을 위한 요약 테이블'''
    # dtypes
    df_dtypes = df.dtypes

    # nunique
    df_nunique = df.nunique()

    # null_values
    df_nan = df.isna().sum()

    val_table = pd.concat([df_dtypes, df_nunique, df_nan], axis=1)
    val_table.columns = ['dtype', 'nunique', 'nan']

    return val_table.reset_index()


train_info = making_val_table(train)
test_info = making_val_table(test)

mergedf = pd.merge(left=train_info, right=test_info, on='index', how='left',
                   suffixes=('_train', '_test')).set_index('index')

# target 변수 지정
y_train = train['ECLO']

# train, test 데이터간 컬럼 동기화
cols = test.columns
X_train = train[cols]
X_test = test


def feat_eng(df):
    # datetime 변환
    df['사고일시'] = pd.to_datetime(df['사고일시'])

    # 월,일,시 컬럼 생성
    df['월'] = df['사고일시'].dt.month
    df['일'] = df['사고일시'].dt.day
    df['시'] = df['사고일시'].dt.hour

    # 불필요 컬럼 제거
    subs = ['ID', '사고일시', '기상상태', '시군구']
    df = df.drop(subs, axis=1)

    # one-hot encoding 실시
    df = pd.get_dummies(df)

    return df


X_train_eng = feat_eng(X_train)
X_test_eng = feat_eng(X_test)


X_train, X_val, y_train, y_val = train_test_split(
    X_train_eng, y_train, test_size=0.2, random_state=42)


rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

y_train_pred = rf.predict(X_train)
y_val_pred = rf.predict(X_val)


rmsle_train = np.sqrt(mean_squared_log_error(y_train, y_train_pred))
rmsle_val = np.sqrt(mean_squared_log_error(y_val, y_val_pred))


print('train rmsle :', rmsle_train)
print('val rmsle :', rmsle_val)


y_pred = rf.predict(X_test_eng)


ss['ECLO'] = y_pred


OUTPUT_PATH = './out'
ss.to_csv(f'{OUTPUT_PATH}submission.csv', index=False)
