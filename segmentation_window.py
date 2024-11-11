import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Importar módulos personalizados
from YOLOv8ObjectDetector import YOLOv8ObjectDetector
from ImageProcessor import ImageProcessor
from YOLOv8BBOX import YOLOv8BBOX
from MergeDF import MergeDF


class ConsoleRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.console_output = ""

    def write(self, message):
        self.console_output += message
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()

    def flush(self):
        pass

    def export_to_txt(self, output_path):
        """Guardar la consola en un archivo de texto."""
        with open(output_path, 'w') as f:
            f.write(self.console_output)


class SegmentationWindow:
    def __init__(self, master):
        self.master = master
        self.master.minsize(width=1000, height=600)
        self.master.maxsize(width=1000, height=600)
        self.master.title('Segmentation - AI Analyzer Program')

        # Crear widget de texto para la consola
        self.console_text = ScrolledText(self.master, wrap='word', height=10, width=100)
        self.console_text.pack(padx=10, pady=10)

        # Barra de progreso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.master, variable=self.progress_var, mode='determinate')
        self.progress_bar.pack(pady=5)

        # Panel para gráficas (Tkinter FigureCanvasTkAgg)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Redireccionar salida de consola a Tkinter
        self.redirect_console_to_tkinter()

        # Variables para rutas de archivo
        self.model_path = None
        self.input_folder = None
        self.output_folder = None

        # Variable para el hilo de ejecución
        self.process_thread = None
        self.stop_thread = False  # Variable para detener el hilo

        # Asociar el evento de cerrar la ventana a una función
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

        # Usar after para comenzar el proceso tras abrir la ventana
        self.master.after(100, self.start_process)

    def redirect_console_to_tkinter(self):
        self.console_redirector = ConsoleRedirector(self.console_text)
        sys.stdout = self.console_redirector
        sys.stderr = self.console_redirector

    def on_close(self):
        """Función que se llama cuando el usuario cierra la ventana."""
        if self.process_thread and self.process_thread.is_alive():
            messagebox.showwarning("Proceso activo", "El proceso está en ejecución. Espera a que termine antes de cerrar.")
        else:
            self.master.destroy()

    def common_file_input(self):
        """Método para seleccionar archivos y carpetas."""
        self.model_path = filedialog.askopenfilename(title="Select YOLO Model File", filetypes=[("YOLO Model", "*.pt")])
        if not self.model_path:
            return None, None, None
        
        self.input_folder = filedialog.askdirectory(title="Select Input Folder")
        if not self.input_folder:
            return None, None, None
        
        self.output_folder = filedialog.askdirectory(title="Select Output Folder for Predictions")
        if not self.output_folder:
            return None, None, None

        return self.model_path, self.input_folder, self.output_folder

    def display_image_on_canvas(self, img):
        """Función para mostrar una imagen en el canvas de matplotlib."""
        self.fig.clear()
        new_ax = self.fig.add_subplot(111)
        new_ax.imshow(img)
        self.canvas.draw()

    def run_all_functions(self):
        """Función que ejecuta todas las tareas en secuencia."""
        model_path, input_folder, output_folder = self.common_file_input()
        if model_path is None:
            return

        total_steps = 4  # Progreso dividido en 4 pasos
        current_step = 0

        # Detección de objetos
        if not self.stop_thread:
            detector = YOLOv8ObjectDetector(model_path, input_folder, output_folder, self)
            detector.predict_and_save(input_folder)
            print("Predictions completed.", flush=True)
            current_step += 1
            self.progress_var.set((current_step / total_steps) * 100)
            self.master.update_idletasks()

        # Procesar imágenes para máscaras
        if not self.stop_thread:
            image_processor = ImageProcessor(model_path, input_folder, output_folder, self)
            image_processor.process_images()
            print("Masks processing completed.", flush=True)
            current_step += 1
            self.progress_var.set((current_step / total_steps) * 100)
            self.master.update_idletasks()

        # Detección de cajas delimitadoras
        if not self.stop_thread:
            bbox_predictor = YOLOv8BBOX(model_path, input_folder, output_folder, self)
            bbox_predictor.predict_and_save_bbox(input_folder)
            print("Bounding box predictions completed.", flush=True)
            current_step += 1
            self.progress_var.set((current_step / total_steps) * 100)
            self.master.update_idletasks()

        # Fusionar archivos CSV
        if not self.stop_thread:
            bbox_csv_filename = 'all_bbox_data.csv'
            masks_csv_filename = 'all_contour_areas.csv'
            bbox_csv_path = os.path.join(output_folder, bbox_csv_filename)
            masks_csv_path = os.path.join(output_folder, masks_csv_filename)
            output_csv_path = os.path.join(output_folder, 'merged_dataframe.csv')
            
            merger = MergeDF(bbox_csv_path, masks_csv_path, output_csv_path)
            merged_df = merger.merge_csv_files()
            print("CSV files merged successfully.", flush=True)
            current_step += 1
            self.progress_var.set((current_step / total_steps) * 100)
            self.master.update_idletasks()

        messagebox.showinfo("Finished", "All functions completed!")

        # Preguntar si se desea exportar el log
        self.ask_export_log()

        # Preguntar si se desea reiniciar
        self.ask_restart()

    def ask_export_log(self):
        """Preguntar si el usuario quiere exportar la consola a un archivo .txt."""
        save_log = messagebox.askyesno("Export Log", "¿Deseas exportar la salida de la consola a un archivo .txt?")
        if save_log:
            file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
            if file_path:
                self.console_redirector.export_to_txt(file_path)
                messagebox.showinfo("Guardado", "La salida ha sido guardada exitosamente.")

    def ask_restart(self):
        """Preguntar si el usuario quiere reiniciar el proceso."""
        restart = messagebox.askyesno("Reiniciar", "¿Deseas reiniciar el proceso?")
        if restart:
            # Resetear la consola y las rutas
            self.console_text.delete(1.0, tk.END)
            self.model_path = None
            self.input_folder = None
            self.output_folder = None
            
            #Limpiar el canvas
            self.fig.clear()  # Limpia la figura actual
            self.canvas.draw()  # Redibuja el canvas vacío
            self.start_process()  # Reiniciar el proceso completo
        else:
            self.master.destroy()

    def run_all_functions_threaded(self):
        """Iniciar `run_all_functions` en un hilo separado."""
        if self.process_thread and self.process_thread.is_alive():
            messagebox.showinfo("Proceso en curso", "El proceso ya está en ejecución.")
        else:
            self.stop_thread = False  # Reiniciar el indicador de detención
            self.process_thread = threading.Thread(target=self.run_all_functions)
            self.process_thread.start()

    def start_process(self):
        """Iniciar el proceso de selección de archivos y ejecución en un hilo."""
        # Usar after para asegurarse de que la ventana está visible antes de pedir los archivos
        self.master.after(100, self.run_all_functions_threaded)
