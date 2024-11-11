import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from YOLOv8ObjectDetector import YOLOv8ObjectDetector  # Asegúrate de que este import sea correcto

class ImageProcessor:
    def __init__(self, model_path, input_folder, output_folder, main_window):
        """
        Initialize the ImageProcessor.

        Args:
            model_path (str): Path to the YOLOv8 model checkpoint file.
            input_folder (str): Path to the folder containing input images.
            output_folder (str): Path to the folder where masks will be saved.
        """
        self.model_path = model_path
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.main_window = main_window
        self.model = YOLOv8ObjectDetector(self.model_path, self.input_folder, self.output_folder, self.main_window)
        self.area_data = []  # Inicializar la lista de datos de área

    def process_images(self):
        """
        Process each image in the input folder and generate masks.

        Returns:
            None
        """
        if not self.input_folder or not self.output_folder:
            raise ValueError("Input and output folders must be specified.")

        total_images = len([f for f in os.listdir(self.input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])

        for i, image_file in enumerate(os.listdir(self.input_folder)):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                # Inicializar datos de área para la imagen actual
                area_data = []

                image_path = os.path.join(self.input_folder, image_file)
                original_image = Image.open(image_path)
                resized_image = original_image.resize((800, 608))
                resized_image_array = np.array(resized_image)
                resized_image_array = cv2.cvtColor(resized_image_array, cv2.COLOR_RGBA2RGB)  # Convertir a RGB

                new_results = self.model.model.predict(resized_image_array, conf=0.3)

                if not new_results:  # Verificar si new_results está vacío o es None
                    print(f"No results for {image_file}, skipping...")
                    continue

                new_result = new_results[0]

                if new_result.masks is None:
                    print(f"No masks found for {image_file}")
                    continue

                # Inicializar un array para almacenar las máscaras
                all_masks = np.zeros_like(resized_image_array[:, :, 0], dtype=np.uint8)

                for idx, mask in enumerate(new_result.masks.data.cpu().numpy()):
                    binary_mask = (mask > 0).astype(np.uint8) * 255

                    # Guardar la máscara binaria para cada máscara por separado
                    output_image_path = os.path.join(self.output_folder, f"{image_file.split('.')[0]}_mask_{idx}.png")
                    cv2.imwrite(output_image_path, binary_mask)  # Cambié plt.imsave a cv2.imwrite
                    print(f"Saved the result image as {output_image_path}")

                    # Calcular el área de contorno para la máscara actual
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    total_contour_area_mask = 0  # Inicializar el área de contorno total para la máscara actual
                    for contour in contours:
                        contour_area = cv2.contourArea(contour)
                        total_contour_area_mask += contour_area

                    # Añadir los datos de área de contorno a la lista para la máscara actual
                    area_data.append({'Mask': f"{image_file.split('.')[0]}_mask_{idx}.png",
                                     'Total_Contour_Area': total_contour_area_mask})

                    # Actualizar el array all_masks con la máscara actual
                    all_masks += binary_mask

                # Guardar la imagen combinada de máscaras
                output_image_path = os.path.join(self.output_folder, f"{image_file.split('.')[0]}_masks.png")
                cv2.imwrite(output_image_path, all_masks)  # Cambié plt.imsave a cv2.imwrite
                print(f"Saved the result image as {output_image_path}")

                # Calcular el área de contorno para la máscara combinada
                contours_combined, _ = cv2.findContours(all_masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                total_contour_area_combined = 0  # Inicializar el área de contorno total para la máscara combinada
                for contour_combined in contours_combined:
                    contour_area_combined = cv2.contourArea(contour_combined)
                    total_contour_area_combined += contour_area_combined

                # Extender el atributo de la clase con los datos de área de la imagen actual
                self.area_data.extend(area_data)

                # Mostrar la máscara combinada en el canvas de la ventana principal de Tkinter
                self.main_window.display_image_on_canvas(all_masks)

                # Actualizar la barra de progreso
                progress_value = (i + 1) / total_images * 100
                self.main_window.progress_var.set(progress_value)
                self.main_window.master.update_idletasks()

        # Guardar los datos de área en un archivo CSV al final
        self.save_contour_areas()

    def save_contour_areas(self):
        """
        Save the area data to a CSV file.
        """
        # Convertir la lista de datos de área extendida a un DataFrame
        df = pd.DataFrame(self.area_data)
        csv_path = os.path.join(self.output_folder, "all_contour_areas.csv")
        df.to_csv(csv_path, index=False)
        print(df)
        print(f"Saved CSV file with all contour areas: {csv_path}")
