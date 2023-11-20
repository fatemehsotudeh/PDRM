import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, \
    roc_curve, auc


def split_dataset(df):
    target_features = ['MCQ160L', 'MCQ220']
    y1 = df['MCQ160L']
    y2 = df['MCQ220']
    X = df.drop(columns=target_features)

    return X, y1, y2


def split_train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def standardize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(multi_class='ovr', max_iter=1000)
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    return model.predict(X_test)


def save_model(model, path):
    joblib.dump(model, path)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='micro',
                                                                     zero_division='warn')

    return accuracy, classification_rep, conf_matrix, precision, recall, f1_score
