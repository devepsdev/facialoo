# Facialoo

Aplicativo de reconocimiento facial con machine learning e inteligencia artificial escrito en Python. Utiliza OpenCV para la detección y reconocimiento de rostros mediante el algoritmo EigenFace, con una interfaz gráfica moderna construida con CustomTkinter.

## Funcionalidades

- **Captura de rostros**: Graba imágenes faciales desde la cámara web y las almacena asociadas a un nombre.
- **Entrenamiento del modelo**: Entrena un reconocedor EigenFace con las imágenes capturadas.
- **Detección en tiempo real**: Detecta rostros en el feed de la cámara y los marca con un recuadro.
- **Reconocimiento en tiempo real**: Identifica personas previamente registradas usando el modelo entrenado.

## Requisitos

- Python 3.8+
- Cámara web

## Instalación

1. Clona el repositorio:

```bash
git clone https://github.com/devepsdev/facialoo.git
cd facialoo
```

1. Crea y activa un entorno virtual:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

1. Instala las dependencias:

```bash
pip install opencv-contrib-python imutils numpy Pillow customtkinter
```

## Uso

Ejecuta la interfaz gráfica:

```bash
python facialoo/gui.py
```

### Flujo de trabajo

1. **Capturar**: Escribe un nombre en el campo de texto y presiona "Capturar". La aplicación grabará 351 imágenes del rostro detectado.
2. **Entrenar**: Presiona "Entrenar" para generar el modelo de reconocimiento con todas las personas capturadas.
3. **Detectar**: Presiona "Detectar" para ver la detección de rostros en tiempo real (sin identificar).
4. **Reconocer**: Presiona "Reconocer" para identificar personas en tiempo real usando el modelo entrenado.

## Estructura del proyecto

```Estructura
facialoo/
├── facialoo/
│   ├── gui.py              # Interfaz gráfica principal
│   ├── captura.py           # Script de captura (standalone)
│   ├── entrenamiento.py     # Script de entrenamiento (standalone)
│   ├── reconocimiento.py    # Script de reconocimiento (standalone)
│   ├── prueba.py            # Script de pruebas
│   └── Data/                # Imágenes capturadas (no incluido en git)
├── .gitignore
└── README.md
```

## Tecnologías

- **OpenCV** - Visión por computadora y procesamiento de imágenes
- **EigenFaceRecognizer** - Algoritmo de reconocimiento facial
- **CustomTkinter** - Interfaz gráfica moderna con tema oscuro
- **Pillow** - Manejo de imágenes para la GUI
- **imutils** - Utilidades para redimensionamiento de frames

---

## 👨‍💻 Autor

**DevEps** - Desarrollador Full Stack

- GitHub: [github.com/devepsdev](https://github.com/devepsdev)
- Portfolio: [deveps.ddns.net](https://deveps.ddns.net)
- Email: devepsdev@gmail.com
- LinkedIn: [www.linkedin.com/in/enrique-perez-sanchez](https://www.linkedin.com/in/enrique-perez-sanchez/)

---

⭐ ¡Dale una estrella si el proyecto te ha resultado útil!
