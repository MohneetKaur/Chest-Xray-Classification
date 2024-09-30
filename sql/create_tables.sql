CREATE TABLE patients (
    patient_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255),
    age INT,
    gender VARCHAR(10),
    medical_history TEXT
);

CREATE TABLE xray_images (
    image_id INT PRIMARY KEY AUTO_INCREMENT,
    patient_id INT,
    image_url VARCHAR(255),
    diagnosis VARCHAR(50),
    confidence FLOAT,
    date_uploaded DATE,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);
