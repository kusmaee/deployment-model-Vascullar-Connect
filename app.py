import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

data = pd.read_csv('heart.csv')

feature_columns = []

# Numeric columns
for header in ['Age', 'RestingBP', 'Cholesterol',
               'MaxHR', 'Oldpeak']:
  feature_columns.append(tf.feature_column.numeric_column(header))

# Indicator Columns
data['ChestPainType'] = data['ChestPainType'].apply(str)
cpt = tf.feature_column.categorical_column_with_vocabulary_list(
    'ChestPainType', ['TA', 'ATA', 'NAP', 'ASY']
)
cpt_one_hot = tf.feature_column.indicator_column(cpt)
feature_columns.append(cpt_one_hot)

data['Sex'] = data['Sex'].apply(str)
sex = tf.feature_column.categorical_column_with_vocabulary_list(
    'Sex', ['M', 'F']
)
sex_one_hot = tf.feature_column.indicator_column(sex)
feature_columns.append(sex_one_hot)

data['RestingECG'] = data['RestingECG'].apply(str)
ecg = tf.feature_column.categorical_column_with_vocabulary_list(
    'RestingECG', ['Normal', 'ST', 'LVH']
)
ecg_one_hot = tf.feature_column.indicator_column(ecg)
feature_columns.append(ecg_one_hot)

data['ExerciseAngina'] = data['ExerciseAngina'].apply(str)
angina = tf.feature_column.categorical_column_with_vocabulary_list(
    'ExerciseAngina', ['Yes', 'No']
)
angina_one_hot = tf.feature_column.indicator_column(angina)
feature_columns.append(angina_one_hot)

data['ST_Slope'] = data['ST_Slope'].apply(str)
slope = tf.feature_column.categorical_column_with_vocabulary_list(
    'ST_Slope', ['Up', 'Flat', 'Down']
)
slope_one_hot = tf.feature_column.indicator_column(slope)
feature_columns.append(slope_one_hot)

def create_dataset(dataframe, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('HeartDisease')
  return tf.data.Dataset.from_tensor_slices((dict(dataframe), labels)) \
          .shuffle(buffer_size=len(dataframe)) \
          .batch(batch_size)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

train, test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)

train_ds = create_dataset(train)
test_ds = create_dataset(test)

from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import tensorflow as tf
from tensorflow import keras
import numpy as np

app = Flask(__name__)

def create_model():
    model = tf.keras.Sequential([
      tf.keras.layers.DenseFeatures(feature_columns=feature_columns),
      tf.keras.layers.Dense(units=512, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(units=256, activation='relu'),
      tf.keras.layers.Dense(units=128, activation='relu'),
      tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    model.fit(
      train_ds,
      validation_data=test_ds,
      epochs=100,
      use_multiprocessing=True
      )
    return model
model = create_model()
model.built = True
model.load_weights('model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        input_data = {
            'Age': np.array([int(data['Age'])]),
            'Sex': np.array([str(data['Sex'])]),
            'ChestPainType': np.array([str(data['ChestPainType'])]),
            'RestingBP': np.array([int(data['RestingBP'])]),
            'Cholesterol': np.array([int(data['Cholesterol'])]),
            'FastingBS': np.array([int(data['FastingBS'])]),
            'RestingECG': np.array([str(data['RestingECG'])]),
            'MaxHR': np.array([int(data['MaxHR'])]),
            'ExerciseAngina': np.array([str(data['ExerciseAngina'])]),
            'Oldpeak': np.array([int(data['OldPeak'])]),
            'ST_Slope': np.array([str(data['ST_Slope'])])
        }

        prediction = model.predict(input_data)

        prediction = (prediction > 0.5).astype(int)  # Assuming binary classification with sigmoid activation

        return jsonify({'prediction': int(prediction[0][0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
