import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib

class EyeDetectionApp:
    def __init__(self, root, video_source=0):
        self.root = root
        self.root.title("Drowsiness Detection App")

        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(root, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_start = tk.Button(root, text="Start", command=self.start)
        self.btn_start.pack(pady=10)

        self.btn_stop = tk.Button(root, text="Stop", command=self.stop)
        self.btn_stop.pack(pady=5)

        self.detect = dlib.get_frontal_face_detector()
        self.predict = dlib.shape_predictor("/Users/harshareddy/VsCode/Pythonn/NULLCLASS/NULL_ASSIGNMENT/Detect_dowsiness/shape_predictor_68_face_landmarks.dat")

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

        self.thresh = 0.25
        self.frame_check = 20
        self.smoothing_factor = 0.2

        self.is_running = False
        self.update_thread = None

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def start(self):
        self.is_running = True
        self.vid = cv2.VideoCapture(self.video_source)
        self.update_thread = threading.Thread(target=self.update)
        self.update_thread.start()

    def stop(self):
        self.is_running = False
        if self.update_thread:
            self.update_thread.join()

    def on_close(self):
        self.is_running = False
        self.vid.release() 
        self.root.destroy()

    def update(self):
        ear = 0  
        while self.is_running:
            ret, frame = self.vid.read()

            if ret:
                frame = cv2.flip(frame, 1)  
                ear = self.update_ear(frame, ear)
                drowsy = self.detect_drowsiness(ear)
                text = "Drowsy" if drowsy else ""
                self.draw_text(frame, text)

               
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.canvas.photo = photo  

            self.root.update()  

        self.vid.release()  

    def update_ear(self, frame, ear):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        subjects = self.detect(gray, 0)

        for subject in subjects:
            shape = self.predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = self.smoothing_factor * ear + (1 - self.smoothing_factor) * ((leftEAR + rightEAR) / 2.0)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        return ear

    def detect_drowsiness(self, ear):
        if ear < self.thresh:
            return True  
        return False  

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def draw_text(self, frame, text):
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

if __name__ == "__main__":
    root = tk.Tk()
    app = EyeDetectionApp(root)
    root.mainloop()
