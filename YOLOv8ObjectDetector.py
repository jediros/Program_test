import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO  # Asegúrate de tener la importación correcta para YOLO

class YOLOv8ObjectDetector:
    def __init__(self, model_path, input_folder, output_folder, main_window):
        """
        Inicializa el detector de objetos YOLOv8.

        Args:
            model_path (str): Ruta al archivo del modelo YOLOv8 (default: 'last.pt').
            output_folder (str): Carpeta donde se guardarán las imágenes predichas (default: 'predicted').
            main_window (tk.Tk): Referencia a la ventana principal de Tkinter para mostrar gráficos y actualizar progreso.
        """
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.main_window = main_window

    def get_total_images(self, image_folder):
        """
        Cuenta el número de archivos de imagen en la carpeta especificada.

        Args:
            image_folder (str): Ruta a la carpeta que contiene las imágenes de entrada.

        Returns:
            int: El número de imágenes en la carpeta.
        """
        supported_formats = ('.jpg', '.jpeg', '.png')  # Formatos de imagen soportados
        image_files = [f for f in os.listdir(image_folder) if f.endswith(supported_formats)]
        return len(image_files)

    def predict_and_save(self, image_folder):
        """
        Predice objetos en las imágenes de la carpeta especificada y guarda los resultados.

        Args:
            image_folder (str): Ruta a la carpeta que contiene las imágenes de entrada.
        """
        if not self.model_path or not os.path.exists(self.model_path):
            raise ValueError("Ruta inválida para el archivo del modelo YOLO.")

        if not os.path.exists(image_folder):
            raise ValueError("Ruta inválida para la carpeta de entrada.")
        
        total_images = self.get_total_images(image_folder)  # Obtener el número total de imágenes

        if total_images == 0:
            print("No se encontraron imágenes en la carpeta.")
            return

        for i, image_file in enumerate(os.listdir(image_folder)):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_folder, image_file)
                original_image = Image.open(image_path)
                resized_image = original_image.resize((800, 608))
                resized_image_array = np.array(resized_image)
                
                # Realizar la predicción
                new_results = self.model.predict(resized_image_array, conf=0.3)
                new_result_array = new_results[0].plot()
                
                total_time = sum(new_results[0].speed.values())

                # Información sobre la inferencia
                inference_info = f"{image_file}: {new_results[0].orig_shape[1]}x{new_results[0].orig_shape[0]} " \
                                 f"{len(new_results[0].boxes)} Embarcaciones, {total_time * 1000:.1f}ms\n"
                print(inference_info)
                
                # Guardar la imagen con la predicción
                output_image_path = os.path.join(self.output_folder, f"{image_file.split('.')[0]}_predicted.png")
                plt.imsave(output_image_path, new_result_array)
                print(f"Guardada la imagen resultado como {output_image_path}")
                
                # Mostrar la imagen en el canvas de Tkinter
                self.main_window.display_image_on_canvas(new_result_array)

                # Actualizar la barra de progreso dinámicamente
                progress_value = (i + 1) / total_images * 100
                self.main_window.progress_var.set(progress_value)
                self.main_window.master.update_idletasks()

        print("Todas las predicciones han sido completadas.")
