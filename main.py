import os
import datetime  # For timestamping saved images

# --- Configuration for TensorFlow ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import pickle

# --- Sound Handling ---
try:
    import winsound
    has_sound = True
except ImportError:
    has_sound = False

# --- Global Variables ---
filename = None
classifier = None

# --- Logic Functions ---
def beep():
    if has_sound:
        frequency = 2500  
        duration = 500  
        winsound.Beep(frequency, duration)

names = ['Accident Occurred', 'No Accident Occurred']

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    log_message(f"Dataset loaded from: {filename}")

def trainCNN():
    global classifier
    global filename
    
    if not os.path.exists('model'):
        os.makedirs('model')

    if filename is None:
        messagebox.showerror("Error", "Please upload dataset first!")
        return

    BATCH_SIZE = 64
    IMG_SIZE = (120, 120)

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    log_message("Loading Data... Please wait.")
    main.update() 

    train_generator = train_datagen.flow_from_directory(
        filename,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        filename,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    if os.path.exists('model/model.json') and os.path.exists('model/model_weights.h5'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/model_weights.h5")
        classifier.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        
        if os.path.exists('model/history.pckl'):
            with open('model/history.pckl', 'rb') as f:
                data = pickle.load(f)
            accuracy = data['accuracy'][-1] * 100
            log_message(f"Model Loaded. Prediction Accuracy: {accuracy:.2f}%")
        else:
             log_message("Model Loaded Successfully.")
    else:
        classifier = Sequential()
        classifier.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(120,120,3)))
        classifier.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Dropout(0.25))
        classifier.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Dropout(0.25))
        classifier.add(Flatten())
        classifier.add(Dense(1024, activation='relu'))
        classifier.add(Dropout(0.5))
        classifier.add(Dense(2, activation='softmax'))
        
        classifier.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        
        steps_per_epoch = train_generator.samples // BATCH_SIZE
        validation_steps = validation_generator.samples // BATCH_SIZE
        if steps_per_epoch == 0: steps_per_epoch = 1
        if validation_steps == 0: validation_steps = 1

        log_message("Training Started... Check console for progress.")
        main.update()

        models_info = classifier.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=40,
            validation_data=validation_generator,
            validation_steps=validation_steps
        )

        classifier.save_weights('model/model_weights.h5')
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
            
        with open('model/history.pckl', 'wb') as f:
            pickle.dump(models_info.history, f)
        
        accuracy = models_info.history['accuracy'][-1] * 100
        log_message(f"Training Complete. Accuracy: {accuracy:.2f}%")

def webcamPredict():
    global classifier
    if classifier is None:
        messagebox.showerror("Error", "Please train or load the model first!")
        return
        
    videofile = askopenfilename(initialdir = "videos")
    if not videofile:
        return
        
    # Ensure directory for saved accidents exists
    if not os.path.exists('detected_accidents'):
        os.makedirs('detected_accidents')

    log_message(f"Analyzing video: {videofile}")
    video = cv2.VideoCapture(videofile)
    
    frame_count = 0 
    accident_detected_flag = False  # Track if ANY accident happens
    
    while(video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            # Preprocess frame
            img = cv2.resize(frame, (120,120))
            img = img.reshape(1, 120,120, 3)
            img = np.array(img, dtype='float32') / 255
            
            # Prediction
            predict = classifier.predict(img)
            class_index = np.argmax(predict)
            confidence = predict[0][class_index] * 100
            result = names[class_index]
            
            # Prepare display text
            label = f"{result}: {confidence:.2f}%"
            
            # Set color: Red for accident, Green for safe
            color = (0, 0, 255) if class_index == 0 else (0, 255, 0)
            
            cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Accident Detection Output", frame)
            
            # Check for Accident (Index 0)
            if class_index == 0:
                accident_detected_flag = True  # Mark that we found an accident
                beep()
                
                # Save screenshot (limit to every 10th frame)
                if frame_count % 10 == 0:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    img_name = f"detected_accidents/accident_{timestamp}.jpg"
                    cv2.imwrite(img_name, frame)
            
            frame_count += 1
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break                
        else:
            break
            
    video.release()
    cv2.destroyAllWindows()
    
    log_message("Video analysis finished.")
    
    # --- FINAL CONCLUSION POPUP ---
    if accident_detected_flag:
        messagebox.showwarning("Final Report", "⚠️ CONCLUSION: Accident DETECTED in this video!")
    else:
        messagebox.showinfo("Final Report", "✅ CONCLUSION: NO Accident detected.")

def graph():
    if not os.path.exists('model/history.pckl'):
        messagebox.showerror("Error", "No training history found!")
        return
        
    with open('model/history.pckl', 'rb') as f:
        data = pickle.load(f)

    plt.figure(figsize=(10,6))
    plt.plot(data['accuracy'], label='Training Accuracy')
    plt.plot(data['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def log_message(msg):
    text_console.insert(END, msg + "\n")
    text_console.see(END)

def close_app():
    main.destroy()

# --- MODERN GUI SETUP ---

# Color Palette (Dark Theme)
COLOR_BG = "#2c3e50"        # Dark Slate Blue (Background)
COLOR_SIDEBAR = "#34495e"   # Lighter Slate (Sidebar)
COLOR_BTN = "#1abc9c"       # Teal (Buttons)
COLOR_BTN_TXT = "white"     # Button Text
COLOR_HEADER = "white"      # Header Text
COLOR_TEXT_BG = "#ecf0f1"   # Light Grey (Text Console)
COLOR_TEXT_FG = "#2c3e50"   # Dark Text (Text Console)

main = tkinter.Tk()
main.title("Accident Detection System")
main.geometry("1100x650")
main.config(bg=COLOR_BG)

# -- Title Bar --
title_frame = Frame(main, bg=COLOR_BG, pady=20)
title_frame.pack(side=TOP, fill=X)
title = Label(title_frame, text='CNN Accident Detection System', font=('Helvetica', 24, 'bold'), bg=COLOR_BG, fg=COLOR_HEADER)
title.pack()

# -- Main Container --
container = Frame(main, bg=COLOR_BG)
container.pack(fill=BOTH, expand=True, padx=20, pady=20)

# -- Left Sidebar (Controls) --
sidebar = Frame(container, bg=COLOR_SIDEBAR, width=300, bd=2, relief=RIDGE)
sidebar.pack(side=LEFT, fill=Y, padx=(0, 20))

# Button Styling Function
def create_btn(parent, text, command):
    return Button(parent, text=text, command=command, 
                  font=('Segoe UI', 12, 'bold'), 
                  bg=COLOR_BTN, fg=COLOR_BTN_TXT, 
                  activebackground="#16a085", activeforeground="white",
                  bd=0, padx=20, pady=10, width=25, cursor="hand2")

# Sidebar Buttons
btn_y = 30
spacer = 70

Label(sidebar, text="Control Panel", font=('Segoe UI', 16), bg=COLOR_SIDEBAR, fg="white").place(x=60, y=20)

btn1 = create_btn(sidebar, "1. Upload Dataset", uploadDataset)
btn1.place(x=25, y=80)

btn2 = create_btn(sidebar, "2. Train/Load Model", trainCNN)
btn2.place(x=25, y=80 + spacer)

btn3 = create_btn(sidebar, "3. Detect in Video", webcamPredict)
btn3.place(x=25, y=80 + spacer*2)

btn4 = create_btn(sidebar, "4. View Accuracy Graph", graph)
btn4.place(x=25, y=80 + spacer*3)

btn_exit = Button(sidebar, text="Exit Application", command=close_app, 
                  font=('Segoe UI', 12, 'bold'), bg="#e74c3c", fg="white", 
                  bd=0, padx=20, pady=10, width=25, cursor="hand2")
btn_exit.place(x=25, y=500)

# -- Right Side (Output Area) --
content_frame = Frame(container, bg=COLOR_BG)
content_frame.pack(side=LEFT, fill=BOTH, expand=True)

# Path Label
Label(content_frame, text="Current Dataset Path:", font=('Segoe UI', 12), bg=COLOR_BG, fg="white").pack(anchor=W)
pathlabel = Label(content_frame, text="No dataset selected", font=('Segoe UI', 10, 'italic'), bg="#34495e", fg="#bdc3c7", width=60, anchor=W, padx=10, pady=5)
pathlabel.pack(anchor=W, pady=(5, 20), fill=X)

# Console Log
Label(content_frame, text="System Log & Predictions:", font=('Segoe UI', 12), bg=COLOR_BG, fg="white").pack(anchor=W)

text_frame = Frame(content_frame)
text_frame.pack(fill=BOTH, expand=True)

scrollbar = Scrollbar(text_frame)
scrollbar.pack(side=RIGHT, fill=Y)

text_console = Text(text_frame, height=15, font=('Consolas', 11), bg=COLOR_TEXT_BG, fg=COLOR_TEXT_FG, yscrollcommand=scrollbar.set, bd=0, padx=10, pady=10)
text_console.pack(side=LEFT, fill=BOTH, expand=True)
scrollbar.config(command=text_console.yview)

# Start App
main.mainloop()