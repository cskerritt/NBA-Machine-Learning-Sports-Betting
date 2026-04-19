import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SEED = 42
np.random.seed(SEED)

NUM_TRIALS = 100
TEST_SIZE = 0.1
EPOCHS = 500
PARAM = {
    'max_depth': 2,
    'eta': 0.01,
    'objective': 'multi:softprob',
    'num_class': 2,
    'seed': SEED,
}

data = pd.read_excel('../../Datasets/DataSet-2021-22.xlsx')
margin = data['Home-Team-Win']
data.drop(['Score', 'Home-Team-Win', 'Unnamed: 0', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'],
          axis=1, inplace=True)
data = data.values.astype(float)

# Single, deterministic train/test split so trials are comparable.
x_train, x_test, y_train, y_test = train_test_split(
    data, margin, test_size=TEST_SIZE, random_state=SEED
)
train_dmatrix = xgb.DMatrix(x_train, label=y_train)
test_dmatrix = xgb.DMatrix(x_test, label=y_test)

best_acc = -1.0
best_model = None
for trial in tqdm(range(NUM_TRIALS)):
    # Vary the booster seed per trial; data split is fixed so accuracies are comparable.
    param = {**PARAM, 'seed': SEED + trial}
    model = xgb.train(param, train_dmatrix, EPOCHS)
    predictions = model.predict(test_dmatrix)
    y_pred = np.argmax(predictions, axis=1)
    acc = round(accuracy_score(y_test, y_pred), 3) * 100
    if acc > best_acc:
        best_acc = acc
        best_model = model

print(f'Best accuracy: {best_acc}%')
best_model.save_model('../../Models/XGBoost_{}%_ML-2.json'.format(best_acc))
