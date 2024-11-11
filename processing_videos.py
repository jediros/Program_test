#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 09:27:18 2024

@author: jedirosero
"""

import os
import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox

class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video File AI Detection")
        self.root.geometry("400x500")

        # Variables para almacenar las rutas seleccionadas
        self.video_path = None
        self.model_path = None
        self.confidence_threshold = tk.DoubleVar(value=0.3)  # Valor por defecto
        self.resize_factor = tk.IntVar(value=1)  # Valor por defecto de resize

        # Crear interfaz
        self.create_widgets()

    def create_widgets(self):
        """Crear la interfaz gráfica."""
        # Botón para seleccionar video
        tk.Button(self.root, text="Select Video File", command=self.select_video_file).pack(pady=10)
        self.video_label = tk.Label(self.root, text="No video selected")
        self.video_label.pack()

        # Botón para seleccionar el modelo .pt
        tk.Button(self.root, text="Select Model (.pt) File", command=self.select_model_file).pack(pady=10)
        self.model_label = tk.Label(self.root, text="No model selected")
        self.model_label.pack()

        # Slider para ajustar el confidence threshold
        tk.Label(self.root, text="Confidence Threshold:").pack(pady=10)
        self.confidence_slider = tk.Scale(self.root, from_=0, to=1, orient=tk.HORIZONTAL, resolution=0.01,
                                           variable=self.confidence_threshold)
        self.confidence_slider.pack()

        # Slider para el resize factor
        tk.Label(self.root, text="Resize Factor:").pack(pady=10)
        self.resize_slider = tk.Scale(self.root, from_=1, to=4, orient=tk.HORIZONTAL, resolution=1,
                                       variable=self.resize_factor)
        self.resize_slider.pack()

        # Botón para procesar un único video
        tk.Button(self.root, text="Process Single Video", command=self.process_single_video).pack(pady=10)

        # Botón para procesar múltiples videos
        tk.Button(self.root, text="Process Multiple Videos", command=self.process_multiple_videos).pack(pady=10)

        # Etiqueta para mostrar la ruta de salida
        self.output_label = tk.Label(self.root, text="Output Path")
        self.output_label.pack(pady=20)

    def select_video_file(self):
        """Abrir el diálogo para seleccionar un archivo de video."""
        try:
            video_path = filedialog.askopenfilename(
                title="Select a video file",
                filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi")]
            )

            if video_path:
                self.video_path = video_path
                self.video_label.config(text=f"Selected: {os.path.basename(video_path)}")
            else:
                self.video_label.config(text="No video selected")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def select_model_file(self):
        """Abrir el diálogo para seleccionar un archivo de modelo .pt."""
        try:
            model_path = filedialog.askopenfilename(
                title="Select a model file", 
                filetypes=[("PyTorch model files", "*.pt")]
            )
            if model_path:
                self.model_path = model_path
                self.model_label.config(text=f"Selected: {os.path.basename(model_path)}")
            else:
                self.model_label.config(text="No model selected")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def process_single_video(self):
        """Procesar un único video."""
        if not self.video_path or not self.model_path:
            messagebox.showerror("Error", "Please select both a video file and a model file.")
            return

        processor = VideoProcessor(model_path=self.model_path, confidence_threshold=self.confidence_threshold.get(),
                                   resize_factor=self.resize_factor.get(), output_label=self.output_label, root=self.root)
        processor.process_single_video(self.video_path)

    def process_multiple_videos(self):
        """Procesar múltiples videos desde una carpeta."""
        video_directory = filedialog.askdirectory(title="Select Folder with Videos")
        if not video_directory or not self.model_path:
            messagebox.showerror("Error", "Please select both a folder with videos and a model file.")
            return

        processor = VideoProcessor(model_path=self.model_path, confidence_threshold=self.confidence_threshold.get(),
                                   resize_factor=self.resize_factor.get(), output_label=self.output_label, root=self.root)
        processor.process_videos(video_directory)

class VideoProcessor:
    def __init__(self, model_path, confidence_threshold=0.3, resize_factor=1, output_label=None, root=None):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.resize_factor = resize_factor
        self.output_label = output_label  # Para actualizar la ruta de salida en la interfaz
        self.stop_processing = False  # Variable para parar el procesamiento en múltiples videos
        self.skip_current_video = False  # Variable para saltar al siguiente video
        self.root = root  # Referencia a la ventana principal de tkinter

    def process_single_video(self, video_path):
        """Procesar un único video."""
        if not video_path:
            messagebox.showerror("Error", "Please select a video file first")
            return
        output_path = self._process_video(video_path, single_video=True)
        if output_path:
            messagebox.showinfo("Done", f"Video processing complete. Output saved at:\n{output_path}")
            self.output_label.config(text=f"Output saved at: {output_path}")

    def process_videos(self, video_directory):
        """Procesar múltiples videos desde una carpeta."""
        video_files = [f for f in os.listdir(video_directory) if f.endswith(('.mp4', '.avi'))]
        if not video_files:
            messagebox.showerror("Error", "No video files found in the selected directory")
            return

        for video_file in video_files:
            if self.stop_processing:  # Si se activa la señal de parar, salir del bucle
                break
            video_path = os.path.join(video_directory, video_file)
            output_path = self._process_video(video_path, single_video=False)
            if output_path:
                self.output_label.config(text=f"Output saved at: {output_path}")
                self.root.update()  # Forzar actualización de tkinter
        
        if not self.stop_processing:
            messagebox.showinfo("Done", f"Video processing complete. Outputs saved in: {video_directory}")
        else:
            messagebox.showwarning("Processing Stopped", "Video processing was stopped manually.")

        # Restablecer el texto de la etiqueta a "Output Path" después de detener el procesamiento
        self.output_label.config(text="Output Path")
        self.root.update()  # Forzar actualización de tkinter

    def _process_video(self, video_path, single_video=False):
        """Procesar el video con el modelo YOLO."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video file.")
            return

        # Obtener dimensiones del video original
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Redimensionar el tamaño de salida
        output_width = original_width // self.resize_factor
        output_height = original_height // self.resize_factor

        output_file = os.path.splitext(video_path)[0] + '_output.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, fourcc, 30.0, (output_width, output_height))

        ret = True
        while ret:
            ret, frame = cap.read()
            if ret:
                # Detectar con YOLO en cada frame
                results = self.model.track(frame, conf=self.confidence_threshold, persist=True)
                frame_ = results[0].plot()

                # Redimensionar si es necesario
                frame_resized = cv2.resize(frame_, (output_width, output_height))

                # Guardar frame procesado
                out.write(frame_resized)

                # Mostrar el frame procesado
                cv2.imshow('Processed Video', frame_resized)

                # Revisar si se presiona la tecla para parar o saltar el procesamiento
                key = cv2.waitKey(25)
                if key == ord('q'):  # Se presiona 'q'
                    if single_video:
                        print(f"Tecla 'q' presionada, deteniendo el procesamiento del video actual...")
                        # Mostrar el path de salida al detener el video actual
                        self.output_label.config(text=f"Output saved at: {output_file}")
                        self.root.update()  # Forzar la actualización de la ventana
                        break  # Si es un solo video, salir del bucle
                    else:
                        print(f"Tecla 'q' presionada, pasando al siguiente video...")
                        # Mostrar el path de salida al detener el video actual
                        self.output_label.config(text=f"Output saved at: {output_file}")
                        self.root.update()  # Forzar la actualización de la ventana
                        self.skip_current_video = True  # En múltiples videos, saltar al siguiente
                        break
                elif key == ord('e'):  # Se presiona 'e' para detener todo el procesamiento
                    print(f"Tecla 'e' presionada, deteniendo todo el procesamiento de múltiples videos...")
                    self.stop_processing = True  # Detener todos los videos
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        return output_file  # Retornar la ruta del archivo de salida

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessorApp(root)
    root.mainloop()
