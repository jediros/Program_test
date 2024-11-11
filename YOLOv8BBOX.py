from ultralytics import YOLO
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image

class YOLOv8BBOX:
    def __init__(self, model_path, input_folder, output_folder, main_window):
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.main_window = main_window

    def predict_and_save_bbox(self, image_folder):
        if not self.model_path or not os.path.exists(self.model_path):
            raise ValueError("Invalid YOLO model file path.")

        if not os.path.exists(image_folder):
            raise ValueError("Invalid input folder path.")

        total_images = len([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
        bbox_data = []

        for i, image_file in enumerate(os.listdir(image_folder)):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_folder, image_file)
                
                try:
                    original_image = Image.open(image_path)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    continue

                resized_image = original_image.resize((800, 608))
                resized_image_array = np.array(resized_image)

                # Predict and ensure new_results_list is structured correctly
                new_results_list = self.model.predict(resized_image_array, conf=0.3)
                if not new_results_list:
                    print(f"No detections for {image_file}.")
                    continue

                new_result = new_results_list[0]

                if hasattr(new_result, 'boxes'):
                    for box in new_result.boxes.xyxy:
                        bbox_width = (box[2] - box[0]).item()
                        bbox_height = (box[3] - box[1]).item()
                        bbox_data.append({
                            'Image': image_file,
                            'BBox_Width': bbox_width,
                            'BBox_Height': bbox_height
                        })

                    visualized_image = resized_image_array.copy()
                    for idx, box in enumerate(new_result.boxes.xyxy):
                        cv2.rectangle(
                            visualized_image,
                            (int(box[0].item()), int(box[1].item())),
                            (int(box[2].item()), int(box[3].item())),
                            color=(255, 0, 0),  # Red color
                            thickness=2
                        )
                        cv2.putText(
                            visualized_image,
                            str(idx),
                            (int(box[0].item()), int(box[1].item())),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2
                        )

                    # Mostrar la imagen con cuadros delimitadores en el canvas de la ventana principal de Tkinter
                    self.main_window.display_image_on_canvas(cv2.cvtColor(visualized_image, cv2.COLOR_BGR2RGB))

                    output_image_path = os.path.join(self.output_folder, f"{image_file.split('.')[0]}_bbox.png")
                    cv2.imwrite(output_image_path, visualized_image)  # Guardar la imagen visualizada
                    print(f"Saved the result image as {output_image_path}")

                # Update progress bar
                progress_value = (i + 1) / total_images * 100
                self.main_window.progress_var.set(progress_value)
                self.main_window.master.update_idletasks()

        # Save the collected bbox data
        bbox_df = pd.DataFrame(bbox_data)
        csv_path = os.path.join(self.output_folder, "all_bbox_data.csv")
        bbox_df.to_csv(csv_path, index=False)
        print(f"Saved CSV file with all bbox data: {csv_path}")
        print(bbox_df)
