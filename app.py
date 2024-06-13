from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

import pandas as pd
data=pd.read_csv(r"C:\Users\abhil\OneDrive\Desktop\covid\Dataset.csv")
data
le=LabelEncoder()

data=data.drop('idseq_sample_name',axis=1)

data=data.drop('CZB_ID',axis=1)

data['viral_status']=  le.fit_transform(data['viral_status'])
data['sequencing_batch']=le.fit_transform(data['sequencing_batch'])
data['gender']=le.fit_transform(data['gender'])

data['SC2_PCR']=le.fit_transform(data['SC2_PCR'])


X=data.iloc[:,0:5]
X
y=data.iloc[:,-1]
y


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.30,random_state=77)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

model = RandomForestClassifier()
model.fit(X,y)


# Create label encoders for gender, SC2_PCR, and sequencing_batch
gender_le = LabelEncoder()
gender_le.fit(["male", "female", "other"])

sc2_pcr_le = LabelEncoder()
sc2_pcr_le.fit(["positive", "negative"])

sequencing_batch_le = LabelEncoder()
sequencing_batch_le.fit(["SEQ003", "SEQ005"])

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        conn.commit()

        c.close()
        conn.close()

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        c.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
        user = c.fetchone()

        c.close()
        conn.close()

        if user:
            return redirect(url_for('submit_data'))
        else:
            return "Invalid credentials. Please try again."
        
    return render_template('login.html')

@app.route('/submit_data', methods=['GET', 'POST'])
def submit_data():
    if request.method == 'POST':
        sequencing_batch = request.form['sequencing_batch']
        gender = request.form['gender']
        age = request.form['age']
        SC2_PCR = request.form['SC2_PCR']
        SC2_rpm = request.form['SC2_rpm']
        
        # Encode sequencing_batch, gender, and SC2_PCR using LabelEncoder
        sequencing_batch_encoded = sequencing_batch_le.transform([sequencing_batch])[0]
        gender_encoded = gender_le.transform([gender])[0]
        sc2_pcr_encoded = sc2_pcr_le.transform([SC2_PCR])[0]
        
        # Convert input data to array for model prediction
        input_data = np.array([sequencing_batch_encoded, gender_encoded, age, sc2_pcr_encoded, SC2_rpm], dtype=float).reshape(1, -1)

        # Predict using the SVM model
        prediction = model.predict(input_data)[0]

        # Translate the prediction result into a human-readable message
        if prediction == 0:
            prediction_message = "The model prediction is: No Virus"
        elif prediction == 1:
            prediction_message = "The model prediction is: Other Virus"
        elif prediction == 2:
            prediction_message = "The model prediction is: SC2"
        else:
            prediction_message = "The model prediction is: Unknown"

        return render_template('classification_result.html', prediction=prediction_message)

    return render_template('classify_text.html')

if __name__ == '__main__':
    app.run(debug=True)
