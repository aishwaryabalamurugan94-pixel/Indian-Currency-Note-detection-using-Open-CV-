import os
import cv2
from ultralytics import YOLO
import pyttsx3
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import json

# ----------------- Voice Engine -----------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)
def speak(text):
    engine.say(text)
    engine.runAndWait()
def speak_async(text):
    threading.Thread(target=speak, args=(text,), daemon=True).start()

# ----------------- Load YOLO Model -----------------
def load_model(model_folder):
    weights_path = None
    for file in os.listdir(model_folder):
        if file.endswith(".pt") and ("best" in file or "last" in file):
            weights_path = os.path.join(model_folder, file)
            break
    if weights_path is None:
        raise FileNotFoundError("No .pt model found in the specified folder.")
    print(f"Loading model: {weights_path}")
    return YOLO(weights_path)

model_folder = r"C:\Users\HP\Documents\my\vs\Cdataset\My First Project.v8-roboflow-instant-6--eval-.yolov8\runs\detect\train\weights"
model = load_model(model_folder)

# ----------------- User Database -----------------
users_file = "users.json"
if not os.path.exists(users_file):
    with open(users_file, "w") as f:
        json.dump({}, f)

def save_user(username, password):
    with open(users_file, "r") as f:
        users = json.load(f)
    users[username] = password
    with open(users_file, "w") as f:
        json.dump(users, f)

def validate_user(username, password):
    with open(users_file, "r") as f:
        users = json.load(f)
    return users.get(username) == password

# ----------------- App Setup -----------------
root = tk.Tk()
root.title("Indian Currency Detector")
root.geometry("900x750")
root.resizable(False, False)

# ----------------- Navbar -----------------
def create_navbar(parent):
    navbar = tk.Frame(parent, bg="#2c3e50", height=50)
    navbar.pack(side="top", fill="x")
    tk.Button(navbar, text="Home", bg="#2c3e50", fg="white", bd=0,
              command=lambda: show_frame("home")).pack(side="left", padx=20)
    tk.Button(navbar, text="Login/Signup", bg="#2c3e50", fg="white", bd=0,
              command=lambda: show_frame("auth")).pack(side="left", padx=20)
    tk.Button(navbar, text="About", bg="#2c3e50", fg="white", bd=0,
              command=lambda: show_frame("about")).pack(side="left", padx=20)

# ----------------- Multi-frame -----------------
frames = {}
def show_frame(name):
    for frame in frames.values():
        frame.pack_forget()
    frames[name].pack(fill="both", expand=True)

# ----------------- Home / Detector -----------------
home_frame = tk.Frame(root)
frames["home"] = home_frame
create_navbar(home_frame)
tk.Label(home_frame, text="Indian Currency Detector", font=("Helvetica", 24)).pack(pady=10)

# Image display
image_label = tk.Label(home_frame, bd=2, relief="sunken")
image_label.pack(pady=10)
result_label = tk.Label(home_frame, text="", font=("Helvetica", 14), justify="left")
result_label.pack(pady=10)

# Webcam control
webcam_running = False
cap = None

# ----------------- Detector Functions -----------------
def detect_currency(frame):
    results = model.predict(frame)
    detected_text = ""
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf*100:.1f}%", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            detected_text += f"{label} ({conf*100:.1f}%)\n"
            speak_async(f"{label} detected")
    return frame, detected_text

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files","*.jpg *.png *.jpeg")])
    if file_path:
        img = cv2.imread(file_path)
        frame, detected_text = detect_currency(img)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((700,400))
        img_tk = ImageTk.PhotoImage(img_pil)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        result_label.config(text=detected_text)

def start_webcam():
    global webcam_running, cap
    if webcam_running: return
    webcam_running = True
    cap = cv2.VideoCapture(0)
    update_frame()

def stop_webcam():
    global webcam_running, cap
    webcam_running = False
    if cap:
        cap.release()
        cap = None

def update_frame():
    global webcam_running, cap
    if webcam_running and cap:
        ret, frame = cap.read()
        if ret:
            frame, detected_text = detect_currency(frame)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb).resize((700,400))
            img_tk = ImageTk.PhotoImage(img_pil)
            image_label.config(image=img_tk)
            image_label.image = img_tk
            result_label.config(text=detected_text)
        root.after(30, update_frame)

def clear_results():
    image_label.config(image="")
    image_label.image = None
    result_label.config(text="")
    stop_webcam()

# Buttons
btn_frame = tk.Frame(home_frame)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Upload Image", width=20, bg="#4CAF50", fg="white", command=upload_image).grid(row=0,column=0,padx=5,pady=5)
tk.Button(btn_frame, text="Start Webcam", width=20, bg="#2196F3", fg="white", command=start_webcam).grid(row=0,column=1,padx=5,pady=5)
tk.Button(btn_frame, text="Stop Webcam", width=20, bg="#f39c12", fg="white", command=stop_webcam).grid(row=1,column=0,padx=5,pady=5)
tk.Button(btn_frame, text="Clear Results", width=20, bg="#9b59b6", fg="white", command=clear_results).grid(row=1,column=1,padx=5,pady=5)
tk.Button(home_frame, text="Reset Spoken Notes", width=20, bg="#8e44ad", fg="white", command=lambda: speak_async("Spoken notes reset")).pack(pady=5)
tk.Button(home_frame, text="Exit", width=20, bg="#e74c3c", fg="white", command=root.destroy).pack(pady=5)

# ----------------- Login/Signup Combined -----------------
auth_frame = tk.Frame(root)
frames["auth"] = auth_frame
create_navbar(auth_frame)

toggle_state = tk.StringVar(value="login")  # login or signup

def toggle_auth():
    if toggle_state.get() == "login":
        toggle_state.set("signup")
        auth_label.config(text="Signup Page")
        auth_button.config(text="Signup", command=signup_action)
        toggle_link.config(text="Already have an account? Login Here")
    else:
        toggle_state.set("login")
        auth_label.config(text="Login Page")
        auth_button.config(text="Login", command=login_action)
        toggle_link.config(text="Don't have an account? Signup Here")

auth_label = tk.Label(auth_frame, text="Login Page", font=("Helvetica",24))
auth_label.pack(pady=50)

tk.Label(auth_frame, text="Username").pack(pady=5)
auth_username = tk.Entry(auth_frame)
auth_username.pack(pady=5)

tk.Label(auth_frame, text="Password").pack(pady=5)
auth_password = tk.Entry(auth_frame, show="*")
auth_password.pack(pady=5)

def login_action():
    user = auth_username.get()
    pwd = auth_password.get()
    if validate_user(user, pwd):
        messagebox.showinfo("Login","Login Successful")
        show_frame("home")
    else:
        messagebox.showerror("Login","Invalid Credentials")

def signup_action():
    user = auth_username.get()
    pwd = auth_password.get()
    if not user or not pwd:
        messagebox.showerror("Signup","Please enter valid credentials")
        return
    save_user(user, pwd)
    messagebox.showinfo("Signup","Account created successfully!")
    toggle_auth()  # switch to login

auth_button = tk.Button(auth_frame, text="Login", width=20, bg="#4CAF50", fg="white", command=login_action)
auth_button.pack(pady=10)

toggle_link = tk.Button(auth_frame, text="Don't have an account? Signup Here", bd=0, fg="blue", command=toggle_auth)
toggle_link.pack()

# ----------------- About Page -----------------
about_frame = tk.Frame(root)
frames["about"] = about_frame
create_navbar(about_frame)
tk.Label(about_frame, text="About This App", font=("Helvetica",24)).pack(pady=50)
tk.Label(about_frame, text="This application helps visually challenged users detect Indian currency using computer vision and voice output.", wraplength=700, justify="center").pack(pady=20)

# ----------------- Show Home -----------------
show_frame("home")
root.mainloop()
