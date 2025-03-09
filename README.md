# 🔥 CodeApha AI Intern Projects 🔥
Welcome👋 to my AI projects repository! This repository contains two AI-based projects:

1. **Fire Detection and Tracking Tool Using YOLOv8** – A real-time object detection tool that identifies and tracks fire using the YOLOv8 model.
2. **Music Generation Using RNNs** – A deep learning-based music generation model that composes piano melodies using Recurrent Neural Networks (RNNs).

Both projects were developed as part of my AI/ML/DL internship at [**CodeAlpha**](https://codealpha.tech).

---

## 🚀 Project 1: Fire Detection and Tracking Tool Using YOLOv8
### Demo 
Watch the demo of the Fire Detection and Tracking Tool on YouTube: [Fire Detection and Tracking Tool Demo](https://youtu.be/Mg5EUHho4-c)

### Overview

This project implements a **Fire Detection and Tracking Tool** using the YOLOv8 (You Only Look Once) object detection algorithm. The tool allows users to perform live detection via webcam or analyze static images for fire detection. The model has been trained using a dataset obtained from **Roboflow**.

### Features

- Real-time fire detection using a webcam.
- Static image fire detection.
- Saves detected outputs for later analysis.

### Requirements

Ensure you have the following installed:

```bash
pip install opencv-python torch torchvision ultralytics
pip install numpy pandas matplotlib  # Optional
```

### Setup & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/AlexKalll/AI-Projects.git
   ```
2. Navigate to the project folder:
   ```bash
   cd AI-Projects/fire-detection-yolo
   ```
3. Download the trained model `best.pt` from your Google Colab environment and place it in the project directory.
4. Run the script:
   ```bash
   python main.py
   ```
5. Choose between:
   - `live` for real-time detection.
   - `static` for analyzing an image.
6. The detected results are saved in the `runs/detects` directory.

### Acknowledgments

Special thanks to **Roboflow** for dataset support.

------------

## 🎶 Project 2: Music Generation Using RNNs

### Overview

This project uses an **LSTM-based Recurrent Neural Network (RNN)** to generate music. Given a sequence of notes, the model learns to predict the next note in the sequence, composing new melodies based on classical piano music.

### Features

- Trains on **MAESTRO Dataset** (1,200 MIDI files).
- Uses **LSTM-based RNN** for music prediction.
- Generates MIDI files as output.

### Requirements

Install dependencies:

```bash
pip install tensorflow pretty_midi pandas numpy matplotlib fluidsynth
```

### Setup & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/AlexKalll/AI-Projects.git
   ```
2. Navigate to the project folder:
   ```bash
   cd AI-Projects/music-generation-rnn
   ```
3. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Mac/Linux
   .\myenv\Scripts\activate   # On Windows
   ```
4. Run the training script:
   ```bash
   python train.py
   ```
5. The generated music file (`output.midi`) will be available in the project directory.

💡 **Tip:** Use [Google Colab](https://colab.research.google.com) for GPU support.

---

## 🎉 Huge Thanks to [CodeAlpha](https://codealpha.tech)!

A heartfelt **thank you** to [**CodeAlpha**](https://codealpha.tech) for providing me with an incredible internship opportunity in `AI/ML/DL`! This experience has been invaluable in my learning journey. 🚀

---

## 📬 Contact Me

Feel free to connect with me:

- [LinkedIn](https://www.linkedin.com/in/kaletsidik-ayalew-mekonnen-34772226b/)
- [Instagram](https://www.instagram.com/kaletsidik.24?igsh=YzljYTk1ODg3Zg==)
- [X (Twitter)](https://x.com/kaletsidike?t=VCe79O084EmE9bM2V5jOIA\&s=09)
- [Telegram](https://t.me/Adragon_de_mello)
- [GitHub](https://github.com/AlexKalll)
- [LeetCode](https://leetcode.com/Alexkal/)

📧 **Email:** [alexkalalw@gmail.com](mailto\:alexkalalw@gmail.com)
        *Kaletsidik Ayalew*
