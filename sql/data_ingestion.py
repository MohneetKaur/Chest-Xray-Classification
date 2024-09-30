import pandas as pd
import mysql.connector

# Load data
patient_data = pd.read_csv('patient_data.csv')
xray_data = pd.read_csv('xray_predictions.csv')

# Connect to MySQL
connection = mysql.connector.connect(
    host='host',
    user='user',
    password='password',
    database='db'
)
cursor = connection.cursor()

# Insert data into SQL tables
for index, row in patient_data.iterrows():
    cursor.execute("INSERT INTO patients (name, age, gender, medical_history) VALUES (%s, %s, %s, %s)",
                   (row['name'], row['age'], row['gender'], row['medical_history']))

for index, row in xray_data.iterrows():
    cursor.execute("INSERT INTO xray_images (patient_id, image_url, diagnosis, confidence, date_uploaded) VALUES (%s, %s, %s, %s, %s)",
                   (row['patient_id'], row['image_url'], row['diagnosis'], row['confidence'], row['date_uploaded']))

connection.commit()
