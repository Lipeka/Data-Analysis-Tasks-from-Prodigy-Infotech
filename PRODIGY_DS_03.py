import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
file_path = "/content/bank-full.csv"
bank_data = pd.read_csv(file_path, sep=';')
print("First few rows of the dataset:")
print(bank_data.head())
print("\nDataset information:")
print(bank_data.info())
print("\nMissing values:")
print(bank_data.isnull().sum())
print("\nTarget variable distribution:")
print(bank_data['y'].value_counts())
label_encoder = LabelEncoder()
bank_data['job'] = label_encoder.fit_transform(bank_data['job'])
bank_data['marital'] = label_encoder.fit_transform(bank_data['marital'])
bank_data['education'] = label_encoder.fit_transform(bank_data['education'])
bank_data['default'] = label_encoder.fit_transform(bank_data['default'])
bank_data['housing'] = label_encoder.fit_transform(bank_data['housing'])
bank_data['loan'] = label_encoder.fit_transform(bank_data['loan'])
bank_data['contact'] = label_encoder.fit_transform(bank_data['contact'])
bank_data['month'] = label_encoder.fit_transform(bank_data['month'])
bank_data['poutcome'] = label_encoder.fit_transform(bank_data['poutcome'])
bank_data['y'] = label_encoder.fit_transform(bank_data['y'])
X = bank_data.drop('y', axis=1)
y = bank_data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
