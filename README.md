# facial-emotion-recognition
The objective of this project is to detect and classify facial expressions from facial images using a deep learning approach. A Convolutional Neural Network (CNN) is employed to classify the input images into one of seven emotion categories: angry, disgust, fear, happy, neutral, sad, and surprise. 

### Emotions Detected
- Angry 😡
- Disgust 🤢
- Fear 😨
- Happy 😊
- Neutral 😐
- Sad 😢
- Surprise 😲

### Project Highlights
- Dataset: FER-2013 style facial expression dataset (grayscale 48×48 images)  
- Model: Custom CNN with 4 convolutional blocks, dropout for regularization  
- Training Accuracy: **~71.5%**  
- Validation Accuracy: **~63.2%**  
- Trained for 100 epochs with Adam optimizer and categorical cross-entropy loss

- ### Project Structure
├── data/               → Dataset (FER-2013 like)
├── models/             → Trained model (.h5)
├── outputs/            → Graphs & predictions
├── src/                → Main Python scripts / Jupyter notebook
├── README.md
└── requirements.txt 

### Tech Stack
- Python, TensorFlow/Keras
- OpenCV, NumPy, Matplotlib, Seaborn

### How to Run
```bash
git clone https://github.com/rv-manjushri/facial-emotion-recognition.git
cd facial-emotion-recognition
pip install -r requirements.txt
jupyter notebook src/Facial_Emotion_Recognition.ipynb

### Accuracy and Loss Plots
<img width="1500" height="500" alt="accuracy" src="https://github.com/user-attachments/assets/f1c7c032-5539-432d-8694-ac355958e231" />

