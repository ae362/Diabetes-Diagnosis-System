import pandas as pd
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


 


# Load the data
data_path = 'Diabetes Dataset/Dataset of Diabetes .csv'
data = pd.read_csv(data_path)

# Data Cleaning
# Convert columns with mixed types to numeric and handle missing values
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(data[numeric_columns].mean())
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Ensure there are no NaN values in the CLASS column
data = data.dropna(subset=['CLASS'])

# Handling CLASS encoding and ensuring continuous range
data['CLASS'] = data['CLASS'].str.strip().astype('category').cat.codes

# Feature Engineering: Adding BMI Category
data['BMI_Category'] = pd.cut(data['BMI'], bins=[0, 18.5, 25, 30, np.inf], labels=[0, 1, 2, 3])
features = data[numeric_columns + ['Gender', 'BMI_Category']]
features = pd.get_dummies(features, columns=['BMI_Category'], drop_first=True)

# Prepare Data for Modeling
labels = data['CLASS']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create an imputer object with a mean filling strategy
imputer = SimpleImputer(strategy='mean')

# Apply imputation to your features
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Apply SMOTE after imputation
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_imputed, y_train)

# Now apply scaling
scaler = StandardScaler()
X_train_smote = scaler.fit_transform(X_train_smote)
X_test = scaler.transform(X_test_imputed)

# Normalize features after resampling
scaler = StandardScaler()
X_train_smote = scaler.fit_transform(X_train_smote)
X_test = scaler.transform(X_test)

# One-hot encode labels
y_train_encoded = tf.keras.utils.to_categorical(y_train_smote, num_classes=len(np.unique(y_train_smote)))
y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=len(np.unique(y_train_smote)))

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_smote.shape[1],), kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(len(np.unique(y_train_smote)), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Re-implement early stopping to be more lenient
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Remove deprecated save_format
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')


# Fit the model
history = model.fit(X_train_smote, y_train_encoded, epochs=100, batch_size=32,
                    validation_split=0.2, callbacks=[early_stopping, model_checkpoint], verbose=1)
# One-hot encode labels
y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=len(np.unique(y_train)))
y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=len(np.unique(y_train)))


# Model Evaluation
y_pred_prob = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_prob, axis=1)

print("Accuracy:", accuracy_score(np.argmax(y_test_encoded, axis=1), y_pred_classes))
print("Confusion Matrix:\n", confusion_matrix(np.argmax(y_test_encoded, axis=1), y_pred_classes))
print("Classification Report:\n", classification_report(np.argmax(y_test_encoded, axis=1), y_pred_classes))

# ROC-AUC Calculation
roc_auc_scores = []
for i in range(len(np.unique(y_train))):
    fpr, tpr, _ = roc_curve(y_test_encoded[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    roc_auc_scores.append(roc_auc)
    print(f"ROC-AUC Score for Class {i}: {roc_auc}")

# Save the trained model in the recommended format
model.save('diabetes_ANN_model.keras', save_format='tf')  # Saving in the TensorFlow native format