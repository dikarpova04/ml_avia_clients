import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


RANDOM_STATE = 42
DATASET_PATH = "https://raw.githubusercontent.com/evgpat/edu_stepik_from_idea_to_mvp/main/datasets/clients.csv"


def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    return df


def preprocess_data(df):
    df = df[['Customer Type', 'Type of Travel', 'Class', 'Departure Delay in Minutes',
           'Arrival Delay in Minutes', 'Inflight wifi service',
           'Online boarding', 'satisfaction']]

    df = df[df['satisfaction'] != '-']

    df.loc[df['Customer Type'].isnull(), 'Customer Type'] = 'Loyal Customer'
    df.loc[df['Type of Travel'].isnull(), 'Type of Travel'] = 'Business travel'

    df.dropna(subset=['Departure Delay in Minutes'], inplace=True)
    df.dropna(subset=['Arrival Delay in Minutes'], inplace=True)

    df.loc[:, 'Inflight wifi service'] = df['Inflight wifi service'].clip(lower=1, upper=5)
    df.dropna(subset=['Inflight wifi service'], inplace=True)

    df.loc[:, 'Online boarding'] = df['Online boarding'].clip(lower=1, upper=5)
    df.dropna(subset=['Online boarding'], inplace=True)

    return df


def prepare_data(df):
    X = df.drop(['satisfaction'], axis=1)
    y = df[['satisfaction']]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=RANDOM_STATE)

    categorical = ['Customer Type', 'Type of Travel', 'Class']
    numeric_features = ['Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Inflight wifi service', 'Online boarding']

    ohe = OneHotEncoder(drop='first', handle_unknown="ignore")
    ohe.fit(X_train[categorical])
    X_train_ohe = ohe.transform(X_train[categorical])
    X_test_ohe = ohe.transform(X_test[categorical])
    X_train_ohe = pd.DataFrame.sparse.from_spmatrix(X_train_ohe, columns=ohe.get_feature_names_out())
    X_test_ohe = pd.DataFrame.sparse.from_spmatrix(X_test_ohe, columns=ohe.get_feature_names_out())

    with open('WEB_APP/ohe.pickle', 'wb') as f:
        pickle.dump(ohe, f)

    scaler = MinMaxScaler()
    scaler.fit(X_train[numeric_features])
    X_train_scaled = pd.DataFrame(scaler.transform(X_train[numeric_features]), columns=numeric_features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test[numeric_features]), columns=numeric_features)

    with open('WEB_APP/scaler.pickle', 'wb') as f:
        pickle.dump(scaler, f)

    X_train_transformed = pd.concat([X_train_ohe, X_train_scaled], axis=1)
    X_test_transformed = pd.concat([X_test_ohe, X_test_scaled], axis=1)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.fit_transform(y_test)

    return X_train_transformed, X_test_transformed, y_train_encoded, y_test_encoded, ohe, scaler


def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    df = load_dataset()
    df_preprocessed = preprocess_data(df)
    X_train, X_test, y_train, y_test, ohe, scaler = prepare_data(df_preprocessed)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print('Accuracy:', accuracy)

    save_model(model, 'WEB_APP/model.pickle')


if __name__ == '__main__':
    main()