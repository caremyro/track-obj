import tkinter as tk
import threading
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import time

model = YOLO("yolov8n.pt")

class VideoApp:
    def __init__(self, root, stream_url):
        self.root = root
        self.root.title("Flux Vidéo Live")
        self.stream_url = stream_url
        
        # Variables essentielles uniquement
        self.running = True
        self.frame = None
        self.frame_lock = threading.Lock()
        self.processed_frame = None
        self.processed_frame_lock = threading.Lock()
        self.rotation_angle = 0
        self.last_process_time = 0
        self.process_interval = 0.3
        self.skip_frames = 4
        self.frame_count = 0
        self.target_fps = 15
        self.last_frame_time = 0
        
        # Interface simplifiée
        self.setup_ui()
        
        # Capture vidéo
        self.cap = cv2.VideoCapture(self.stream_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        
        # Threads
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.process_thread = threading.Thread(target=self.process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        self.update_display()
    
    def setup_ui(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(expand=True, fill=tk.BOTH)
        
        self.label = tk.Label(self.main_frame)
        self.label.pack(expand=True, fill=tk.BOTH)
        
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=5)
        
        self.rotate_left_btn = tk.Button(self.button_frame, text="Rotation -90°", command=self.rotate_left)
        self.rotate_left_btn.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.rotate_right_btn = tk.Button(self.button_frame, text="Rotation +90°", command=self.rotate_right)
        self.rotate_right_btn.pack(side=tk.LEFT, padx=10, pady=5)
    
    def rotate_left(self):
        self.rotation_angle = (self.rotation_angle - 90) % 360

    def rotate_right(self):
        self.rotation_angle = (self.rotation_angle + 90) % 360

    def rotate_frame(self, frame):
        if self.rotation_angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation_angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def capture_frames(self):
        while self.running:
            current_time = time.time()
            if current_time - self.last_frame_time < 1.0 / self.target_fps:
                time.sleep(0.001)
                continue

            ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = self.rotate_frame(frame)
                with self.frame_lock:
                    self.frame = frame
                self.last_frame_time = current_time
                time.sleep(0.01)

    def process_frames(self):
        while self.running:
            self.frame_count += 1
            if self.frame_count % self.skip_frames != 0:
                time.sleep(0.01)
                continue
            
            current_time = time.time()
            if current_time - self.last_process_time < self.process_interval:
                time.sleep(0.01)
                continue
            
            with self.frame_lock:
                if self.frame is None:
                    continue
                frame = self.frame.copy()
            
            height, width = frame.shape[:2]
            if width > 480:
                scale = 480 / width
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            
            results = model(frame, verbose=False, conf=0.4)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{model.names[cls]} {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            with self.processed_frame_lock:
                self.processed_frame = frame
            
            self.last_process_time = current_time

    def update_display(self):
        if self.running:
            display_frame = None
            with self.processed_frame_lock:
                if self.processed_frame is not None:
                    display_frame = self.processed_frame.copy()
            
            if display_frame is None:
                with self.frame_lock:
                    if self.frame is not None:
                        display_frame = self.frame.copy()
            
            if display_frame is not None:
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.label.config(image=imgtk)
                self.label.image = imgtk
        
        self.root.after(50, self.update_display)

    def cleanup(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()

    def stop(self):
        self.cleanup()
        self.root.quit()
        self.root.destroy()

def start_video_app():
    url = url_entry.get()
    if not url.startswith("http"):
        return

    entry_window.destroy()
    root = tk.Tk()
    app = VideoApp(root, url)
    root.protocol("WM_DELETE_WINDOW", app.stop)
    root.mainloop()

entry_window = tk.Tk()
entry_window.title("Connexion au flux vidéo")

tk.Label(entry_window, text="Entrez l'URL du flux vidéo :").pack(pady=5)
url_entry = tk.Entry(entry_window, width=40)
url_entry.pack(padx=10, pady=5)
url_entry.insert(0, "http://192.168.1.70:4747/video")

tk.Button(entry_window, text="Afficher le flux", command=start_video_app).pack(pady=10)

entry_window.mainloop()