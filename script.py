import tkinter as tk #make the app 
import threading #don't block the app when analysing 
import cv2 #recover the video flow 
from PIL import Image, ImageTk #convert cv -> tkinter
from ultralytics import YOLO #vision model 
import queue
import time

model = YOLO("yolov8n.pt")

class VideoApp:
    def __init__(self, root, stream_url):
        self.root = root
        self.root.title("Flux Vidéo Live")
        self.stream_url = stream_url
        self.cap = cv2.VideoCapture(self.stream_url)
        
        # Vérifier si le flux vidéo est ouvert correctement
        if not self.cap.isOpened():
            print(f"Erreur: Impossible d'ouvrir le flux vidéo à l'URL: {stream_url}")
            self.root.destroy()
            return
            
        # Configuration de la capture vidéo pour de meilleures performances
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Réduire la taille du buffer
        
        self.rotation_angle = 0  
        self.label = tk.Label(root)
        self.label.pack(expand=True, fill=tk.BOTH)
        
        self.rotate_left_btn = tk.Button(root, text="Rotation -90°", command=self.rotate_left)
        self.rotate_left_btn.pack(side=tk.LEFT, padx=10, pady=5)
        self.rotate_right_btn = tk.Button(root, text="Rotation +90°", command=self.rotate_right)
        self.rotate_right_btn.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Variables pour le multithreading
        self.running = True
        self.frame_queue = queue.Queue(maxsize=2)  # Limiter la taille de la file d'attente
        self.imgtk = None
        
        # Démarrer les threads
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Démarrer la mise à jour de l'interface
        self.update_display()

    def rotate_left(self):
        self.rotation_angle = (self.rotation_angle - 90) % 360
        print(f"Rotation actuelle : {self.rotation_angle}°")

    def rotate_right(self):
        self.rotation_angle = (self.rotation_angle + 90) % 360
        print(f"Rotation actuelle : {self.rotation_angle}°")

    def rotate_frame(self, frame):
        if self.rotation_angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation_angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def capture_frames(self):
        """Thread pour capturer les frames vidéo"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Erreur de lecture du flux vidéo.")
                time.sleep(0.1)  # Attendre un peu avant de réessayer
                continue
                
            # Appliquer la rotation
            frame = self.rotate_frame(frame)
            
            # Mettre la frame dans la file d'attente, en supprimant l'ancienne si nécessaire
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()  # Supprimer l'ancienne frame
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # Ignorer si la file est pleine
                
            # Limiter la fréquence de capture
            time.sleep(0.01)  # 10ms de délai entre les captures

    def process_frames(self):
        """Thread pour traiter les frames avec YOLO"""
        while self.running:
            try:
                # Récupérer une frame de la file d'attente
                frame = self.frame_queue.get(timeout=0.1)
                
                # Redimensionner l'image pour accélérer le traitement
                height, width = frame.shape[:2]
                if width > 640:  # Limiter la taille maximale
                    scale = 640 / width
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                
                # Utilisation de YOLO pour la détection d'objets
                results = model(frame, verbose=False)  # Désactiver les messages de progression
                
                # Dessiner les boîtes autour des objets détectés
                for result in results[0].boxes:
                    x1, y1, x2, y2 = result.xyxy[0]
                    conf = result.conf[0]
                    cls = result.cls[0]
                    
                    if conf > 0.5:
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', 
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Convertir l'image pour l'affichage
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                self.imgtk = ImageTk.PhotoImage(image=img)
                
            except queue.Empty:
                pass  # Ignorer si la file est vide
            except Exception as e:
                print(f"Erreur lors du traitement: {e}")
                time.sleep(0.1)

    def update_display(self):
        """Mettre à jour l'affichage de l'interface"""
        if self.running and self.imgtk:
            self.label.config(image=self.imgtk)
        self.root.after(50, self.update_display)  # Mise à jour toutes les 50ms

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        # Attendre que les threads se terminent
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1.0)
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1.0)

def start_video_app():
    url = url_entry.get()
    if not url.startswith("http"):
        url_entry.delete(0, tk.END)
        url_entry.insert(0, "URL invalide")
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
url_entry.insert(0, "http://192.168.X.X:4747/video")

tk.Button(entry_window, text="Afficher le flux", command=start_video_app).pack(pady=10)

entry_window.mainloop()