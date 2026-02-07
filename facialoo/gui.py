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

# Colores personalizados
BG_DARK = "#1a1a2e"
BG_CARD = "#16213e"
ACCENT = "#0f3460"
ACCENT_HOVER = "#1a5276"
SUCCESS = "#2ecc71"
DANGER = "#e74c3c"
TEXT_PRIMARY = "#e0e0e0"
TEXT_SECONDARY = "#a0a0a0"


class FacialRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Reconocimiento Facial")
        self.geometry("780x680")
        self.minsize(700, 620)
        self.configure(fg_color=BG_DARK)

        self.state_mode = IDLE
        self.cap = None
        self.running = False
        self.capture_count = 0
        self.face_cascade = cv.CascadeClassifier(CASCADE_PATH)

        self._build_ui()
        self._update_buttons()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # --- Header ---
        header = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=0, height=56)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="  Reconocimiento Facial",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=TEXT_PRIMARY,
        ).pack(side="left", padx=16)

        self.mode_badge = ctk.CTkLabel(
            header,
            text="LISTO",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#fff",
            fg_color=SUCCESS,
            corner_radius=12,
            width=80,
            height=28,
        )
        self.mode_badge.pack(side="right", padx=16)

        # --- Contenedor principal ---
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=12)

        # --- Video card ---
        video_card = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=14)
        video_card.pack(fill="both", expand=True)

        # Canvas para video con bordes redondeados simulados
        video_inner = ctk.CTkFrame(video_card, fg_color="#000", corner_radius=10)
        video_inner.pack(fill="both", expand=True, padx=12, pady=12)

        self._blank = tk.PhotoImage(width=640, height=480)
        self.video_label = tk.Label(video_inner, bg="#000", image=self._blank)
        self.video_label.pack(expand=True)

        # --- Panel de controles ---
        controls = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=14)
        controls.pack(fill="x", pady=(12, 0))

        # Fila nombre
        name_row = ctk.CTkFrame(controls, fg_color="transparent")
        name_row.pack(fill="x", padx=16, pady=(14, 8))

        ctk.CTkLabel(
            name_row,
            text="Nombre",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_SECONDARY,
        ).pack(side="left")

        self.name_entry = ctk.CTkEntry(
            name_row,
            placeholder_text="Escribe el nombre de la persona...",
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(size=13),
        )
        self.name_entry.pack(side="left", fill="x", expand=True, padx=(10, 0))

        # Fila botones
        btn_row = ctk.CTkFrame(controls, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(4, 14))

        btn_style = dict(
            height=38,
            corner_radius=10,
            font=ctk.CTkFont(size=13, weight="bold"),
            border_width=0,
        )

        self.btn_capture = ctk.CTkButton(
            btn_row, text="Capturar", command=self._start_capture,
            fg_color="#2980b9", hover_color="#3498db", **btn_style,
        )
        self.btn_train = ctk.CTkButton(
            btn_row, text="Entrenar", command=self._start_training,
            fg_color="#8e44ad", hover_color="#9b59b6", **btn_style,
        )
        self.btn_detect = ctk.CTkButton(
            btn_row, text="Detectar", command=self._start_detect,
            fg_color="#27ae60", hover_color="#2ecc71", **btn_style,
        )
        self.btn_recognize = ctk.CTkButton(
            btn_row, text="Reconocer", command=self._start_recognize,
            fg_color="#d35400", hover_color="#e67e22", **btn_style,
        )
        self.btn_stop = ctk.CTkButton(
            btn_row, text="Detener", command=self._stop,
            fg_color=DANGER, hover_color="#c0392b", **btn_style,
        )

        for btn in (self.btn_capture, self.btn_train, self.btn_detect, self.btn_recognize, self.btn_stop):
            btn.pack(side="left", expand=True, fill="x", padx=3)

        # --- Barra de progreso (captura) ---
        self.progress_bar = ctk.CTkProgressBar(
            controls, height=6, corner_radius=3,
            fg_color="#2c3e50", progress_color="#3498db",
        )
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=16, pady=(0, 6))
        self.progress_bar.pack_forget()  # oculta por defecto

        # --- Status bar ---
        status_bar = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=0, height=32)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)

        self.status_var = tk.StringVar(value="Listo")
        self.status_label = ctk.CTkLabel(
            status_bar,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY,
            anchor="w",
        )
        self.status_label.pack(side="left", padx=12)

    def _update_buttons(self):
        idle = self.state_mode == IDLE
        normal = "normal"
        disabled = "disabled"
        self.btn_capture.configure(state=normal if idle else disabled)
        self.btn_train.configure(state=normal if idle else disabled)
        self.btn_detect.configure(state=normal if idle else disabled)
        self.btn_recognize.configure(state=normal if idle else disabled)
        self.btn_stop.configure(state=normal if not idle else disabled)

        # Badge
        badge_map = {
            IDLE: ("LISTO", SUCCESS),
            CAPTURE: ("CAPTURA", "#2980b9"),
            DETECT: ("DETECCION", "#27ae60"),
            RECOGNIZE: ("RECONOCE", "#d35400"),
            TRAINING: ("ENTRENA", "#8e44ad"),
        }
        text, color = badge_map.get(self.state_mode, ("LISTO", SUCCESS))
        self.mode_badge.configure(text=text, fg_color=color)

    def _set_status(self, text):
        self.status_var.set(text)

    # -------------------------------------------------------------- Camera
    def _open_camera(self):
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la camara.")
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
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
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
            messagebox.showwarning("Aviso", "Escribe un nombre antes de capturar.")
            return

        person_dir = os.path.join(DATA_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        if not self._open_camera():
            return

        self.state_mode = CAPTURE
        self.capture_count = 0
        self._update_buttons()
        self._set_status(f"Capturando rostros de '{name}'... 0/{MAX_CAPTURES}")

        # Mostrar barra de progreso
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=16, pady=(0, 6))

        threading.Thread(target=self._capture_loop, args=(person_dir,), daemon=True).start()

    def _capture_loop(self, person_dir):
        while self.running and self.capture_count < MAX_CAPTURES:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = imutils.resize(frame, width=640)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                if self.capture_count >= MAX_CAPTURES:
                    break
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = frame[y:y + h, x:x + w]
                roi = cv.resize(roi, (160, 160), interpolation=cv.INTER_CUBIC)
                cv.imwrite(os.path.join(person_dir, f"imagen_{self.capture_count}.jpg"), roi)
                self.capture_count += 1
                progress = self.capture_count / MAX_CAPTURES
                self.after(0, self._update_capture_progress, progress)

            self.after(0, self._show_frame, frame)

        self.after(0, self._on_capture_done)

    def _update_capture_progress(self, progress):
        self.progress_bar.set(progress)
        self._set_status(f"Capturando rostros... {self.capture_count}/{MAX_CAPTURES}")

    def _on_capture_done(self):
        self._release_camera()
        self._clear_video()
        self.state_mode = IDLE
        self._update_buttons()
        self._set_status(f"Captura finalizada - {self.capture_count} imagenes guardadas")
        self.progress_bar.pack_forget()

    # ----------------------------------------------------------- Train
    def _start_training(self):
        if not os.path.isdir(DATA_DIR) or not os.listdir(DATA_DIR):
            messagebox.showwarning("Aviso", "No hay datos en Data/. Captura rostros primero.")
            return

        self.state_mode = TRAINING
        self._update_buttons()
        self._set_status("Entrenando modelo... por favor espere")

        # Mostrar barra indeterminada
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=16, pady=(0, 6))
        self._train_pulse()

        threading.Thread(target=self._train_model, daemon=True).start()

    def _train_pulse(self):
        if self.state_mode != TRAINING:
            return
        cur = self.progress_bar.get()
        nxt = cur + 0.02 if cur < 0.95 else 0.0
        self.progress_bar.set(nxt)
        self.after(80, self._train_pulse)

    def _train_model(self):
        try:
            ids = []
            faces = []
            label = 0
            for folder in sorted(os.listdir(DATA_DIR)):
                folder_path = os.path.join(DATA_DIR, folder)
                if not os.path.isdir(folder_path):
                    continue
                for filename in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, filename)
                    img = cv.imread(img_path, 0)
                    if img is None:
                        continue
                    faces.append(img)
                    ids.append(label)
                label += 1

            recognizer = cv.face.EigenFaceRecognizer_create()
            recognizer.train(faces, np.array(ids))
            recognizer.write(MODEL_PATH)
            self.after(0, self._on_train_done, None)
        except Exception as e:
            self.after(0, self._on_train_done, str(e))

    def _on_train_done(self, error):
        self.state_mode = IDLE
        self._update_buttons()
        self.progress_bar.pack_forget()
        if error:
            self._set_status(f"Error en entrenamiento: {error}")
            messagebox.showerror("Error", error)
        else:
            self._set_status("Entrenamiento completado")

    # ----------------------------------------------------------- Detect
    def _start_detect(self):
        if not self._open_camera():
            return
        self.state_mode = DETECT
        self._update_buttons()
        self._set_status("Deteccion de rostros en tiempo real")
        self._detect_tick()

    def _detect_tick(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if not ret:
            self._stop()
            return
        frame = imutils.resize(frame, width=640)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        self._show_frame(frame)
        self.after(33, self._detect_tick)

    # --------------------------------------------------------- Recognize
    def _start_recognize(self):
        if not os.path.isfile(MODEL_PATH):
            messagebox.showwarning("Aviso", "No se encontro el modelo entrenado. Entrena primero.")
            return

        if not os.path.isdir(DATA_DIR) or not os.listdir(DATA_DIR):
            messagebox.showwarning("Aviso", "No hay datos en Data/.")
            return

        self.recognizer = cv.face.EigenFaceRecognizer_create()
        self.recognizer.read(MODEL_PATH)
        self.label_names = sorted(
            [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        )

        if not self._open_camera():
            return

        self.state_mode = RECOGNIZE
        self._update_buttons()
        self._set_status("Reconocimiento en tiempo real")
        self._recognize_tick()

    def _recognize_tick(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if not ret:
            self._stop()
            return
        frame = imutils.resize(frame, width=640)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi = cv.resize(roi, (160, 160), interpolation=cv.INTER_CUBIC)
            result = self.recognizer.predict(roi)

            if result[1] < 8000:
                name = self.label_names[result[0]] if result[0] < len(self.label_names) else "?"
                cv.putText(frame, name, (x, y - 20), 2, 1.1, (0, 255, 0), 1, cv.LINE_AA)
            else:
                cv.putText(frame, "No encontrado", (x, y - 20), 2, 0.7, (0, 255, 0), 1, cv.LINE_AA)

            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.putText(frame, f"{result}", (x, y - 5), 1, 1.3, (0, 255, 0), 1, cv.LINE_AA)

        self._show_frame(frame)
        self.after(33, self._recognize_tick)

    # ------------------------------------------------------------- Stop
    def _stop(self):
        self._release_camera()
        self._clear_video()
        self.state_mode = IDLE
        self._update_buttons()
        self._set_status("Listo")
        self.progress_bar.pack_forget()

    # --------------------------------------------------------------- Close
    def _on_close(self):
        self._release_camera()
        self.destroy()


if __name__ == "__main__":
    app = FacialRecognitionApp()
    app.mainloop()
