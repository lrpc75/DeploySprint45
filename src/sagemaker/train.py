import argparse
import os
import pandas as pd
import xgboost as xgb

def train(args):
    # Carregar dados de treino e teste
    train_data = pd.read_csv(os.path.join(args.train, 'train.csv'), header=None)
    test_data = pd.read_csv(os.path.join(args.validation, 'test.csv'), header=None)
    
    # Separar características e rótulos
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    # Treinar o modelo XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='multi:softmax',
        num_class=3,
        num_rounds = 100
    )
    model.fit(X_train, y_train)

    # Avaliar o modelo
    accuracy = model.score(X_test, y_test)
    print(f'Acurácia do modelo: {accuracy}')

    # Salvar o modelo treinado
    model.save_model(os.path.join(args.model_dir, 'xgboost-model'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Argumentos do SageMaker
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    args = parser.parse_args()
    train(args)
