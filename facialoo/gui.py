import cv2 as cv
import os
import threading
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
import imutils
import numpy as np
from PIL import Image, ImageTk

# --- Rutas basadas en la ubicación de este archivo ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
MODEL_PATH = os.path.join(BASE_DIR, "EntrenamientoEigenFaceRecognizer.xml")

# Haarcascade portátil (incluido con opencv)
CASCADE_PATH = cv.data.haarcascades + "haarcascade_frontalface_default.xml"

# Estados
IDLE = "IDLE"
CAPTURE = "CAPTURE"
DETECT = "DETECT"
RECOGNIZE = "RECOGNIZE"
TRAINING = "TRAINING"

MAX_CAPTURES = 351

# Tema
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Colores personalizados (Modern Dark Theme)
BG_DARK = "#121212"       # Fondo principal muy oscuro
BG_SIDEBAR = "#1E1E1E"    # Sidebar ligeramente más claro
BG_CARD = "#252526"       # Tarjetas/contenedores
ACCENT_PRIMARY = "#007ACC" # Azul VSCode-like para acciones principales
ACCENT_HOVER = "#005A9E"
SUCCESS = "#4CAF50"       # Verde material
WARNING = "#FFC107"       # Amber
DANGER = "#F44336"        # Rojo material
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#BBBBBB"
BORDER_COLOR = "#333333"

class FacialRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Facialoo - Reconocimiento Facial AI")
        self.geometry("1000x700")
        self.minsize(900, 650)
        self.configure(fg_color=BG_DARK)

        self.state_mode = IDLE
        self.cap = None
        self.running = False
        self.capture_count = 0
        self.face_cascade = cv.CascadeClassifier(CASCADE_PATH)

        # Configurar Grid principal (1x2: Sidebar | Main)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_ui()
        self._update_buttons()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # --- Sidebar (Izquierda) ---
        self.sidebar = ctk.CTkFrame(self, fg_color=BG_SIDEBAR, corner_radius=0, width=280)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(6, weight=1) # Empujar contenido hacia arriba

        # Logo/Título
        self.title_label = ctk.CTkLabel(
            self.sidebar, 
            text="FACIALOO", 
            font=ctk.CTkFont(family="Roboto", size=24, weight="bold"),
            text_color=TEXT_PRIMARY
        )
        self.title_label.grid(row=0, column=0, padx=20, pady=(30, 10), sticky="w")
        
        self.subtitle_label = ctk.CTkLabel(
            self.sidebar, 
            text="AI Facial Recognition", 
            font=ctk.CTkFont(family="Roboto", size=12),
            text_color=TEXT_SECONDARY
        )
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 30), sticky="w")

        # Entrada de Nombre
        self.name_label = ctk.CTkLabel(
            self.sidebar, 
            text="Nombre del Sujeto:", 
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_SECONDARY
        )
        self.name_label.grid(row=2, column=0, padx=20, pady=(0, 5), sticky="w")

        self.name_entry = ctk.CTkEntry(
            self.sidebar,
            placeholder_text="Ej. John Doe",
            height=40,
            corner_radius=8,
            font=ctk.CTkFont(size=14),
            border_color=BORDER_COLOR,
            fg_color=BG_CARD
        )
        self.name_entry.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="ew")

        # Botones de Acción
        self.btn_capture = self._create_sidebar_btn("Capturar Rostro", self._start_capture, icon="📷", row=4)
        self.btn_train = self._create_sidebar_btn("Entrenar Modelo", self._start_training, icon="🧠", row=5)
        
        # Espaciador (Row 6 tiene weight=1)

        # Botones de Inferencia (Detectar/Reconocer)
        self.btn_detect = self._create_sidebar_btn("Detectar", self._start_detect, icon="👁️", row=7, color=SUCCESS)
        self.btn_recognize = self._create_sidebar_btn("Reconocer", self._start_recognize, icon="🔍", row=8, color=ACCENT_PRIMARY)
        
        # Botón Detener
        self.btn_stop = ctk.CTkButton(
            self.sidebar, 
            text="⛔ DETENER", 
            command=self._stop,
            height=45,
            corner_radius=8,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=DANGER,
            hover_color="#D32F2F",
            state="disabled"
        )
        self.btn_stop.grid(row=9, column=0, padx=20, pady=20, sticky="ew")


        # --- Main Content (Derecha) ---
        self.main_content = ctk.CTkFrame(self, fg_color="transparent")
        self.main_content.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_content.grid_rowconfigure(1, weight=1)
        self.main_content.grid_columnconfigure(0, weight=1)

        # Header Superior Main
        self.header_frame = ctk.CTkFrame(self.main_content, fg_color="transparent", height=50)
        self.header_frame.grid(row=0, column=0, sticky="ew", pady=(10, 20))
        
        self.status_label_header = ctk.CTkLabel(
            self.header_frame, 
            text="Sistema Listo", 
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT_PRIMARY
        )
        self.status_label_header.pack(side="left")

        self.mode_badge = ctk.CTkLabel(
            self.header_frame,
            text="IDLE",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#fff",
            fg_color=BORDER_COLOR,
            corner_radius=15,
            width=100,
            height=30,
        )
        self.mode_badge.pack(side="right")

        # Video Frame Container
        self.video_frame = ctk.CTkFrame(self.main_content, fg_color=BG_CARD, corner_radius=20, border_width=2, border_color=BORDER_COLOR)
        self.video_frame.grid(row=1, column=0, sticky="nsew")
        self.video_frame.pack_propagate(False) # Respetar tamaño del grid

        # Canvas para video (centrado)
        self.video_inner = ctk.CTkFrame(self.video_frame, fg_color="#000", corner_radius=15)
        self.video_inner.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.95, relheight=0.95)

        self._blank = tk.PhotoImage(width=1, height=1) # Placeholder invisible
        self.video_label = tk.Label(self.video_inner, bg="#000", image=self._blank)
        self.video_label.pack(expand=True, fill="both")

        # Barra de Progreso (Overlay o debajo del video)
        self.progress_bar = ctk.CTkProgressBar(
            self.main_content, 
            height=10, 
            corner_radius=5,
            progress_color=ACCENT_PRIMARY,
            fg_color=BORDER_COLOR
        )
        self.progress_bar.set(0)
        self.progress_bar.grid(row=2, column=0, sticky="ew", pady=(20, 0))
        self.progress_bar.grid_remove() # Ocultar inicialmente

    def _create_sidebar_btn(self, text, command, icon="", row=0, color="transparent"):
        btn = ctk.CTkButton(
            self.sidebar,
            text=f"  {icon}  {text}",
            command=command,
            height=45,
            corner_radius=8,
            anchor="w",
            font=ctk.CTkFont(size=14),
            fg_color=color if color != "transparent" else "transparent",
            text_color=TEXT_PRIMARY if color != "transparent" else TEXT_SECONDARY,
            border_width=1 if color == "transparent" else 0,
            border_color=BORDER_COLOR,
            hover_color=BG_CARD if color == "transparent" else None
        )
        btn.grid(row=row, column=0, padx=20, pady=5, sticky="ew")
        return btn

    def _update_buttons(self):
        idle = self.state_mode == IDLE
        normal = "normal"
        disabled = "disabled"
        
        # Habilitar/Deshabilitar botones sidebar
        for btn in [self.btn_capture, self.btn_train, self.btn_detect, self.btn_recognize]:
            btn.configure(state=normal if idle else disabled)
        
        # Botón Stop
        self.btn_stop.configure(state=normal if not idle else disabled, fg_color=DANGER if not idle else BG_CARD)

        # Badge y Texto Header
        badge_map = {
            IDLE: ("SISTEMA EN ESPERA", "IDLE", BORDER_COLOR),
            CAPTURE: (f"CAPTURANDO: {self.name_entry.get().upper()}", "CAPTURING", ACCENT_PRIMARY),
            DETECT: ("DETECTANDO ROSTROS...", "DETECTING", SUCCESS),
            RECOGNIZE: ("IDENTIFICANDO USUARIOS...", "RECOGNIZING", WARNING),
            TRAINING: ("ENTRENANDO IA...", "TRAINING", "#9C27B0"),
        }
        status_text, badge_text, badge_color = badge_map.get(self.state_mode, ("DESCONOCIDO", "ERROR", DANGER))
        
        self.status_label_header.configure(text=status_text)
        self.mode_badge.configure(text=badge_text, fg_color=badge_color)

    def _set_status(self, text):
        # Actualizamos el header en lugar de una barra inferior
        self.status_label_header.configure(text=text)

    # -------------------------------------------------------------- Camera
    def _open_camera(self):
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error de Cámara", "No se pudo acceder a la webcam.\nVerifique la conexión.")
            return False
        self.running = True
        return True

    def _release_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ------------------------------------------------------- Video refresh
    def _show_frame(self, frame):
        # Resize inteligente para ajustar al contenedor sin perder aspect ratio
        viewport_width = self.video_inner.winfo_width()
        viewport_height = self.video_inner.winfo_height()
        
        if viewport_width < 10 or viewport_height < 10:
             viewport_width, viewport_height = 640, 480 # Fallback

        h, w = frame.shape[:2]
        
        # Calcular escala manteniendo aspect ratio
        scale = min(viewport_width/w, viewport_height/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv.resize(frame, (new_w, new_h))
        frame_rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
        
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

    def _clear_video(self):
        self.video_label.config(image=self._blank)
        self.video_label.imgtk = None

    # ------------------------------------------------------------ Capture
    def _start_capture(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Falta Información", "Por favor ingrese un nombre para el usuario.")
            return

        person_dir = os.path.join(DATA_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        if not self._open_camera():
            return

        self.state_mode = CAPTURE
        self.capture_count = 0
        self._update_buttons()
        # self._set_status se maneja en _update_buttons pero podemos forzar info extra
        
        # Mostrar barra de progreso
        self.progress_bar.set(0)
        self.progress_bar.grid()

        threading.Thread(target=self._capture_loop, args=(person_dir,), daemon=True).start()

    def _capture_loop(self, person_dir):
        # Frame delay para suavidad vs rendimiento
        while self.running and self.capture_count < MAX_CAPTURES:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Espejo (opcional, suele ser más natural)
            frame = cv.flip(frame, 1)

            # Proceso de detección
            small_frame = imutils.resize(frame, width=640)
            gray = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            # Dibujar en frame original (escalado)
            # Nota: Si redimensionamos `frame` para visualización en _show_frame, 
            # las coordenadas de `faces` (de small_frame) necesitan ajuste si queremos pintar en HR.
            # Simplicidad: usaremos small_frame para visualización también por rendimiento.

            display_frame = small_frame.copy()

            for (x, y, w, h) in faces:
                if self.capture_count >= MAX_CAPTURES:
                    break
                
                # Visual feedback
                cv.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Guardar ROI
                roi = gray[y:y + h, x:x + w]
                roi = cv.resize(roi, (160, 160), interpolation=cv.INTER_CUBIC)
                cv.imwrite(os.path.join(person_dir, f"imagen_{self.capture_count}.jpg"), roi)
                
                self.capture_count += 1
                progress = self.capture_count / MAX_CAPTURES
                self.after(0, self._update_capture_progress, progress)

            self.after(0, self._show_frame, display_frame)

        self.after(0, self._on_capture_done)

    def _update_capture_progress(self, progress):
        self.progress_bar.set(progress)
        
    def _on_capture_done(self):
        self._release_camera()
        self._clear_video()
        self.state_mode = IDLE
        self._update_buttons()
        self.progress_bar.grid_remove()
        messagebox.showinfo("Captura Finalizada", f"Se han guardado {self.capture_count} imágenes para '{self.name_entry.get()}'.")

    # ----------------------------------------------------------- Train
    def _start_training(self):
        if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
            messagebox.showwarning("Sin Datos", "No hay rostros capturados para entrenar.")
            return

        self.state_mode = TRAINING
        self._update_buttons()
        
        self.progress_bar.set(0)
        self.progress_bar.grid()
        self._train_pulse()

        threading.Thread(target=self._train_model, daemon=True).start()

    def _train_pulse(self):
        if self.state_mode != TRAINING:
            return
        cur = self.progress_bar.get()
        nxt = cur + 0.02 if cur < 0.95 else 0.0
        self.progress_bar.set(nxt)
        self.after(50, self._train_pulse)

    def _train_model(self):
        try:
            ids = []
            faces = []
            label = 0
            
            # Obtener lista de carpetas válidas
            people = sorted([p for p in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, p))])
            
            if not people:
                raise ValueError("Directorio Data vacío o sin carpetas válidas.")

            for person_name in people:
                person_path = os.path.join(DATA_DIR, person_name)
                for filename in os.listdir(person_path):
                    img_path = os.path.join(person_path, filename)
                    try:
                        img = cv.imread(img_path, 0)
                        if img is not None:
                            faces.append(img)
                            ids.append(label)
                    except Exception:
                        pass
                label += 1

            if not faces:
                raise ValueError("No se encontraron imágenes válidas para entrenar.")

            recognizer = cv.face.EigenFaceRecognizer_create()
            recognizer.train(faces, np.array(ids))
            recognizer.write(MODEL_PATH)
            self.after(0, self._on_train_done, None)
            
        except Exception as e:
            self.after(0, self._on_train_done, str(e))

    def _on_train_done(self, error):
        self.state_mode = IDLE
        self._update_buttons()
        self.progress_bar.grid_remove()
        if error:
            messagebox.showerror("Error de Entrenamiento", error)
        else:
            messagebox.showinfo("Éxito", "Modelo entrenado correctamente.")

    # ----------------------------------------------------------- Detect
    def _start_detect(self):
        if not self._open_camera():
            return
        self.state_mode = DETECT
        self._update_buttons()
        self._detect_tick()

    def _detect_tick(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if not ret:
            self._stop()
            return

        frame = cv.flip(frame, 1)
        frame = imutils.resize(frame, width=800) # Un poco más resolución para display
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Detección
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Diseño de caja más elegante
            color = (0, 255, 0) # BGR
            # Esquinas redondeadas simuladas (líneas cortas en esquinas)
            l = 30 # longitud de esquina
            t = 2 # grosor
            
            # Top-Left
            cv.line(frame, (x, y), (x + l, y), color, t)
            cv.line(frame, (x, y), (x, y + l), color, t)
            # Top-Right
            cv.line(frame, (x + w, y), (x + w - l, y), color, t)
            cv.line(frame, (x + w, y), (x + w, y + l), color, t)
            # Bottom-Left
            cv.line(frame, (x, y + h), (x + l, y + h), color, t)
            cv.line(frame, (x, y + h), (x, y + h - l), color, t)
            # Bottom-Right
            cv.line(frame, (x + w, y + h), (x + w - l, y + h), color, t)
            cv.line(frame, (x + w, y + h), (x + w, y + h - l), color, t)

        self._show_frame(frame)
        self.after(30, self._detect_tick)

    # --------------------------------------------------------- Recognize
    def _start_recognize(self):
        if not os.path.isfile(MODEL_PATH):
            messagebox.showwarning("Modelo Faltante", "No se encontró el archivo de entrenamiento.\nPor favor entrene el modelo primero.")
            return

        if not os.path.exists(DATA_DIR):
            messagebox.showwarning("Datos Faltantes", "No existe el directorio de datos.")
            return

        # Cargar etiquetas y modelo
        self.label_names = sorted(
            [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        )
        try:
            self.recognizer = cv.face.EigenFaceRecognizer_create()
            self.recognizer.read(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Error al leer modelo", str(e))
            return

        if not self._open_camera():
            return

        self.state_mode = RECOGNIZE
        self._update_buttons()
        self._recognize_tick()

    def _recognize_tick(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if not ret:
            self._stop()
            return
        
        frame = cv.flip(frame, 1)
        frame = imutils.resize(frame, width=800)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi = cv.resize(roi, (160, 160), interpolation=cv.INTER_CUBIC)
            
            # Predicción
            label_id, confidence = self.recognizer.predict(roi)
            
            # Lógica de confianza (EigenFaces: menor es mejor match, usualmente < 5000-8000)
            # Ajustar umbral según necesidad
            threshold = 8000 # Umbral empírico
            
            if confidence < threshold:
                name = self.label_names[label_id] if label_id < len(self.label_names) else "Unknown"
                color = (0, 255, 0)
                status_text = f"{name}"
            else:
                name = "Desconocido"
                color = (0, 0, 255) # Rojo
                status_text = "Desconocido"
            
            # Dibujar caja
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Dibujar etiqueta con fondo
            (text_w, text_h), _ = cv.getTextSize(status_text, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv.rectangle(frame, (x, y - 30), (x + text_w, y), color, -1)
            cv.putText(frame, status_text, (x, y - 6), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Info técnica (opcional)
            # cv.putText(frame, f"Conf: {int(confidence)}", (x, y + h + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        self._show_frame(frame)
        self.after(30, self._recognize_tick)

    # ------------------------------------------------------------- Stop
    def _stop(self):
        self._release_camera()
        self._clear_video()
        self.state_mode = IDLE
        self._update_buttons()
        self.progress_bar.grid_remove()

    # --------------------------------------------------------------- Close
    def _on_close(self):
        self._release_camera()
        self.destroy()

if __name__ == "__main__":
    app = FacialRecognitionApp()
    app.mainloop()

