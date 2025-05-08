from flask import Flask, render_template, request, redirect, session, send_file
import cv2
import os
import numpy as np
import face_recognition
import pyttsx3
import speech_recognition as sr
import smtplib
import webbrowser
import threading
import pickle
import queue
from email.message import EmailMessage

app = Flask(__name__)
app.secret_key = 'your_secret_key'
face_dir = 'face_data'
encodings_file = 'face_encodings.pkl'

if not os.path.exists(face_dir):
    os.makedirs(face_dir)

# Text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('voice', engine.getProperty('voices')[1].id)
listener = sr.Recognizer()
speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# Email setup
EMAIL_ADDRESS = "praveenkalyan8800@gmail.com"
EMAIL_PASSWORD = "cdvb ahkc gepp pqkg"

def send_email(subject, recipient, body):
    try:
        msg = EmailMessage()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.set_content(body)

        print("Attempting to send email...")
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        say("Email has been sent successfully.")
        webbrowser.open("https://mail.google.com/mail/u/0/#sent")
    except Exception as e:
        say("Failed to send email.")
        print(f"Email sending error: {e}")

def say(text):
    speech_queue.put(text)

def assistant_listener():
    try:
        with sr.Microphone() as source:
            listener.adjust_for_ambient_noise(source, duration=1)
            print("Listening for command...")
            voice = listener.listen(source, timeout=5)
            command = listener.recognize_google(voice, language="en-in").lower()
            print(f"Command recognized: {command}")
            return command
    except Exception as e:
        print("Error in listening:", e)
        return "error"

def voice_controlled_email(command):
    if command == "open mail":
        say("Opening Gmail")
        webbrowser.open("https://mail.google.com/mail/u/0/#inbox")
    
    elif command == "send mail":
        # Prompt for the subject
        say("What is the subject of your email?")
        subject = assistant_listener()
        if subject == "error":
            say("I could not hear the subject. Please try again.")
            return

        # Predefined recipient and body for now
        recipient = "vtu20161@veltech.edu.in"  # Can be dynamically
        body = "This is a test email body."  # Can be adjusted dynamically

        # Send the email with the provided subject
        send_email(subject, recipient, body)
        say("Your email has been sent.")
        webbrowser.open("https://mail.google.com/mail/u/0/#sent")
        
    else:
        say("Command not recognized. Please try again.")
        print(f"Unrecognized command: {command}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_id = request.form['user_id']
        user_folder = os.path.join(face_dir, user_id)
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
        
        session['user_id'] = user_id
        return redirect('/capture_images')
    return render_template('register.html')

@app.route('/capture_images')
def capture_images():
    if 'user_id' not in session:
        return redirect('/')
    
    user_id = session['user_id']
    user_folder = os.path.join(face_dir, user_id)
    cap = cv2.VideoCapture(0)
    count = 0
    face_encodings = []
    
    while count < 20:
        ret, frame = cap.read()
        if not ret:
            break
        
        face_locations = face_recognition.face_locations(frame)
        encodings = face_recognition.face_encodings(frame, face_locations)
        if encodings:
            img_path = os.path.join(user_folder, f'img_{count + 1}.jpg')
            cv2.imwrite(img_path, frame)
            face_encodings.append(encodings[0])
            count += 1
        
        cv2.imshow('Capturing Images', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    with open(encodings_file, "ab") as f:
        pickle.dump({'user_id': user_id, 'encoding': np.mean(face_encodings, axis=0)}, f)
    
    return redirect('/train_model')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            say("Failed to capture image.")
            return redirect('/login')
        
        face_encodings = face_recognition.face_encodings(frame)
        if not face_encodings:
            say("No face detected.")
            return redirect('/login')
        
        if os.path.exists(encodings_file):
            with open(encodings_file, "rb") as f:
                known_faces = []
                while True:
                    try:
                        known_faces.append(pickle.load(f))
                    except EOFError:
                        break
        else:
            say("No registered faces found.")
            return redirect('/login')
        
        for known_face in known_faces:
            if isinstance(known_face, dict) and 'encoding' in known_face:
                matches = face_recognition.compare_faces([known_face['encoding']], face_encodings[0])
                if True in matches:
                    session['user_id'] = known_face['user_id']
                    say("Login successful.")
                    return redirect('/speech')
        
        say("Face not recognized.")
        return redirect('/login')
    
    return render_template('login.html')

@app.route('/speech', methods=['GET', 'POST'])
def speech():
    if 'user_id' not in session:
        return redirect('/')
    
    if request.method == 'POST':
        def listen_for_command():
            command = assistant_listener()
            if command != "error":
                voice_controlled_email(command)

        # Start the voice recognition in a separate thread
        thread = threading.Thread(target=listen_for_command)
        thread.start()
        
        # Give feedback to the user while waiting for voice command
        return render_template('speech.html', message="Listening for command...")

    return render_template('speech.html', message="Click to start voice command.")

if __name__ == '__main__':
    app.run(debug=True)
