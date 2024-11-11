import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import threading

class CameraDetection:
    def __init__(self, root, width=1920, height=1080, confidence_threshold=0.3):
        self.root = root
        self.model = None  # El modelo se cargará después
        self.width = width
        self.height = height
        self.confidence_threshold = confidence_threshold

        # Variables de control para la grabación
        self.recording = False
        self.video_writer = None
        self.file_path = ""

        # Variables para mostrar en la GUI
        self.model_label_var = tk.StringVar(value="No model selected")
        self.resolution_label_var = tk.StringVar(value="No set webcam resolution")

        # Crear la ventana
        self.create_camera_detection_window()

    def create_camera_detection_window(self):
        """Crea la interfaz para la detección de cámara."""
        self.window = tk.Toplevel(self.root)
        self.window.title("Video Camera AI Detection")

        # Botón para seleccionar el modelo .pt
        model_button = tk.Button(self.window, text="Select Model (.pt) File", command=self.select_model_file)
        model_button.pack(pady=10)

        # Etiqueta para mostrar el modelo seleccionado
        self.model_label = tk.Label(self.window, textvariable=self.model_label_var)
        self.model_label.pack(pady=5)

        # Botón para usar la webcam
        usb_button = tk.Button(self.window, text="Use USB Webcam", command=self.use_webcam)
        usb_button.pack(pady=10)

        # Botón para configurar la resolución de la webcam
        self.resolution_button = tk.Button(self.window, text="Set Webcam Resolution", command=self.set_resolution)
        self.resolution_button.pack(pady=10)

        # Etiqueta para mostrar la resolución seleccionada
        self.resolution_label = tk.Label(self.window, textvariable=self.resolution_label_var)
        self.resolution_label.pack(pady=5)

        # Etiqueta para el umbral de confianza
        confidence_label = tk.Label(self.window, text="Confidence Threshold")
        confidence_label.pack(pady=5)

        # Entrada para el umbral de confianza
        self.threshold_entry = tk.Entry(self.window)
        self.threshold_entry.insert(0, str(self.confidence_threshold))
        self.threshold_entry.pack(pady=10)

        # Botones de grabación (start/stop)
        self.start_button = tk.Button(self.window, text="Start Recording", command=self.start_recording, state=tk.DISABLED)
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(self.window, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

    def select_model_file(self):
        """Selecciona el archivo del modelo .pt y muestra el nombre del archivo en la ventana."""
        model_path = filedialog.askopenfilename(
            title="Select a model file",
            filetypes=[("PyTorch model files", "*.pt")]
        )
        if model_path:
            self.model = YOLO(model_path)
            model_name = model_path.split('/')[-1]  # Extraer solo el nombre del archivo
            self.model_label_var.set(f"Model loaded: {model_name}")  # Actualizar la etiqueta en la GUI
        else:
            self.model_label_var.set("No model selected")

    def use_webcam(self):
        """Inicia la captura de la webcam."""
        if not self.model:
            messagebox.showerror("Error", "Please select a model file first.")
            return

        index = simpledialog.askinteger("Camera Index", "Enter the USB camera index (0, 1, 2, etc.):", minvalue=0, maxvalue=10)
        if index is None:
            return

        try:
            confidence_threshold = float(self.threshold_entry.get())
            if not (0 <= confidence_threshold <= 1):
                raise ValueError("Threshold must be between 0 and 1")
        except ValueError:
            messagebox.showerror("Error", "Invalid confidence threshold.")
            return

        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open USB webcam")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Crear un hilo para la webcam para evitar bloquear la GUI
        self.webcam_thread = threading.Thread(target=self.webcam_stream, args=(confidence_threshold,))
        self.webcam_thread.start()

    def webcam_stream(self, confidence_threshold):
        """Hilo que maneja el stream de la webcam y la detección en tiempo real."""
        self.start_button.config(state=tk.NORMAL)  # Activar botón de grabación después de iniciar el streaming

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Realizar la detección
            results = self.model.track(frame, conf=confidence_threshold, persist=True)
            frame_ = results[0].plot()

            # Mostrar la imagen en una ventana OpenCV
            cv2.imshow('Webcam Live', frame_)

            # Grabar si está habilitado
            if self.recording and self.video_writer is not None:
                self.video_writer.write(frame)

            # Salir al presionar 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def set_resolution(self):
        """Establece una resolución predeterminada o personalizada para la webcam."""
        resolutions = {
            "1080p (1920x1080)": (1920, 1080),
            "720p (1280x720)": (1280, 720),
            "480p (640x480)": (640, 480),
            "360p (640x360)": (640, 360),
            "Custom Resolution": "custom"  # Opción para resolución personalizada
        }

        resolution_window = tk.Toplevel(self.root)
        resolution_window.title("Set Webcam Resolution")

        instructions_label = tk.Label(resolution_window, text="Select a resolution:")
        instructions_label.pack(pady=5)

        resolution_var = tk.StringVar(resolution_window)
        resolution_var.set("1080p (1920x1080)")

        resolution_menu = tk.OptionMenu(resolution_window, resolution_var, *resolutions.keys())
        resolution_menu.pack(pady=5)

        def apply_resolution():
            selected_resolution = resolution_var.get()

            if selected_resolution == "Custom Resolution":
                resolution_input = simpledialog.askstring("Custom Resolution", "Enter resolution (Width x Height):")
                if resolution_input is None:
                    return

                try:
                    width, height = map(int, resolution_input.split('x'))
                    self.width, self.height = width, height
                    self.resolution_label_var.set(f"Custom resolution set: {self.width}x{self.height}")
                    messagebox.showinfo("Resolution Set", f"Custom resolution set to {self.width}x{self.height}")
                except ValueError:
                    self.resolution_label_var.set("No set custom resolution")
                    messagebox.showerror("Error", "Invalid resolution format.")
            else:
                self.width, self.height = resolutions[selected_resolution]
                self.resolution_label_var.set(f"Resolution set: {self.width}x{self.height}")
                messagebox.showinfo("Resolution Set", f"Webcam resolution set to {self.width}x{self.height}")

            resolution_window.destroy()

        apply_btn = tk.Button(resolution_window, text="Apply", command=apply_resolution)
        apply_btn.pack(pady=5)

    def start_recording(self):
        """Inicia la grabación de video."""
        if not self.recording:
            # Iniciar la grabación sin pedir el nombre del archivo
            self.recording = True
            self.start_button.config(state=tk.DISABLED)  # Desactivar botón de grabar
            self.stop_button.config(state=tk.NORMAL)  # Activar botón de detener grabación
            messagebox.showinfo("Recording Started", "Video recording has started.")

    def stop_recording(self):
        """Detiene la grabación de video."""
        if self.recording:
            self.recording = False

            # Pedir el nombre del archivo solo al detener la grabación
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
            self.file_path = filedialog.asksaveasfilename(
                title="Save Video",
                defaultextension=".mp4",
                filetypes=[("MP4 files", "*.mp4")])
            if not self.file_path:
                return  # Si el usuario cancela, no hacer nada

            self.video_writer = cv2.VideoWriter(self.file_path, fourcc, 20.0, (self.width, self.height))

            # Mensaje de parada
            messagebox.showinfo("Recording Stopped", f"Video recording has stopped. File saved as: {self.file_path}")

            # Limpiar estado de botones
            self.start_button.config(state=tk.NORMAL)  # Activar botón de grabar
            self.stop_button.config(state=tk.DISABLED)  # Desactivar botón de detener grabación

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraDetection(root)
    root.mainloop()
