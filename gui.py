#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 12:51:07 2024

@author: jedirosero
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Importar ventanas adicionales y módulos de procesamiento
from segmentation_window import SegmentationWindow
from camera_detection import CameraDetection
from processing_videos import VideoProcessorApp  # Cambiado para usar la nueva clase VideoProcessorApp

class MainWindow:
    def __init__(self, master):
        self.master = master
        self.master.minsize(width=1000, height=600)
        self.master.maxsize(width=1000, height=600)
        self.master.title('AI - ANALIZER PROGRAM')

        # Barra de progreso
        #self.progress_var = tk.DoubleVar()
        #self.progress_bar = ttk.Progressbar(self.master, variable=self.progress_var, mode='determinate')
        #self.progress_bar.pack(pady=5)

        # Panel para gráficas
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Crear menú y botones
        self.create_menu()

        # Asociar el evento de cerrar la ventana a una función
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_menu(self):
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Exit", command=self.close_program)

        # Menú AI-Analysis
        image_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="AI-ANALYSIS", menu=image_menu)

        # Submenú Detection con opciones
        detection_menu = tk.Menu(image_menu, tearoff=0)
        image_menu.add_cascade(label="Detection", menu=detection_menu)
        detection_menu.add_command(label="Video file", command=self.open_file_detection_window)
        detection_menu.add_command(label="Video camera", command=self.process_camera_detection)

        # Opción de segmentación
        image_menu.add_command(label="Segmentation", command=self.open_segmentation_window)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.about)

    def close_program(self):
        """Función para cerrar el programa."""
        self.master.destroy()

    def on_close(self):
        """Función que se llama cuando el usuario cierra la ventana."""
        os._exit(0)  # Forzar el cierre inmediato del programa

    def open_segmentation_window(self):
        """Abrir la ventana de segmentación en una nueva ventana."""
        new_window = tk.Toplevel(self.master)
        SegmentationWindow(new_window)

    def open_file_detection_window(self):
        """Abrir una nueva ventana para la detección basada en archivo de video."""
        new_window = tk.Toplevel(self.master)
        new_window.title("Video File Detection")
        new_window.geometry("400x500")

        # Crear la clase VideoProcessorApp en esta ventana
        app = VideoProcessorApp(new_window)

    def process_camera_detection(self):
        """Abre la ventana de detección de cámara."""
        CameraDetection(self.master)  # Se pasa la ventana principal como root


    def about(self):
        """Mostrar información del programa."""
        win = tk.Toplevel(self.master)
        win.title('About')

        text_frame = tk.Frame(win)
        text_frame.pack(fill=tk.BOTH, expand=True)

        T = tk.Text(text_frame, wrap="none", height=15, width=100)
        T.pack(side=tk.LEFT, fill=tk.Y)

        scrollbar = tk.Scrollbar(text_frame, command=T.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        T.config(yscrollcommand=scrollbar.set)

        quote = """LOAD PREDICTOR - AI - Analyzer Program' v1.0
        IMPORTANT: For more information or program changes please contact the author."""

        T.insert(tk.END, quote)
        T.tag_configure("justify", justify="left")
        T.insert(tk.END, "\n", "justify")

        tk.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()
