# <p align="center">SignSync: Sign Language Translation for Video Conferencing</p>

<hr />

<p align="center">
<img width="1280" alt="thumbnail" src="https://github.com/user-attachments/assets/628363f2-507d-4921-91e9-2fa932d54f6c" />
</p>

<hr />

## 🧩 Overview
SignSync is **the first ever** innovative real-time sign language translation tool integrated directly into the open-source video conferencing platform Jitsi. It bridges the communication gap between the Deaf and Hard of Hearing (D/HH) community and the general public by making sign language visible, understandable, and inclusive during online meetings.

Imagine you have lost your voice—how will you express your thoughts and communicate with those around you? This is a daily dilemma for the D/HH community.

While sign language is widely used within the D/HH community, over 95% of the hearing population is unfamiliar with it, leading to communication barriers in schools, workplaces, and social settings. 

As part of an inclusive future, SignSync empowers the D/HH community by allowing their expressions to be translated in real time and understood by everyone in a video call.

While tools like Zoom and Microsoft Teams excel in text and voice communication, they lack built-in support for real-time sign language interpretation. That’s the gap we’re aiming to bridge.

## 🧩 Demo

https://github.com/user-attachments/assets/82eed43a-424c-4849-935b-f63aba4ae0bd



## 🚀 Features
🤟 Real-time Sign Language Recognition
Translates sign gestures into readable text using a deep learning model (CNN-based).

📡 Seamless Jitsi Integration
Embedded into the Jitsi interface with no need for separate software.

🌍 Privacy-Conscious
All processing can run locally without uploading sensitive data to external servers.

🖥️ Cross-Platform
Works on modern web browsers across operating systems.

## 🛠️ Tech Stack
Frontend: JavaScript, React (Jitsi Meet UI)

Backend/ML: Python, OpenCV, PyTorch/TensorFlow

Model: Mediapipe for landmakr extraction + CNN-based Sign Language Translation for fingerspelling

## 🧠 How It Works
The webcam captures sign gestures from the user.

Mediapipe hands is called to extract hand landmarks.

Extracted landmarks are being detected by model hosted on frontend device.

Calling Gemini API to separate glued text and correct wrong words.

Display translated text on subtitle overlay.

## 🙌 Why It Matters
By making video communication accessible to the D/HH community, SignSync doesn’t just enable conversation—it promotes inclusion, equity, and visibility.

