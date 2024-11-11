# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:08:27 2024

@author: Jedi Rosero
"""

"""yolo_app/
│
├── main.py                   # Archivo principal que ejecuta la aplicación
├── gui.py                    # Manejo de la interfaz gráfica (Tkinter)
├── segmentation_window.py
    ├── YOLOv8ObjectDetector.py    
    ├── ImageProcessor.py         # Funciones relacionadas con el procesamiento de imágenes
    ├── YOLOv8BBOX.py             
    ├── MergeDF.py                # Newly created file
├──processing_videos.py
├──camera_detection.py

"""
#main.py 
import tkinter as tk
from gui import MainWindow


#gui.py
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Importar ventanas adicionales y módulos de procesamiento
from segmentation_window import SegmentationWindow
from camera_detection import CameraDetection
from processing_videos import VideoProcessorApp  # Cambiado para usar la nueva clase VideoProcessorApp

#segmentation_window.py
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

#├── YOLOv8ObjectDetector.py   

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO  # Asegúrate de tener la importación correcta para YOLO
 
#├── ImageProcessor.py  
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from YOLOv8ObjectDetector import YOLOv8ObjectDetector  # Asegúrate de que este import sea correcto       # Funciones relacionadas con el procesamiento de imágenes


#├── YOLOv8BBOX.py  

from ultralytics import YOLO
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
           
#├── MergeDF.py    
import pandas as pd



#processing_videos.py
import os
import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox


#camera_detection.py
import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import threading



import cv2
print(cv2.__version__)

import numpy as np
print(np.__version__)

import PIL
print(PIL.__version__)

import matplotlib
print(matplotlib.__version__)

import pandas as pd
print(pd.__version__)

import ultralytics
print(ultralytics.__version__)

import torch
print(torch.__version__)


"""

opencv-python==4.7.0  # OpenCV para Python
numpy==1.26.4  # NumPy
Pillow==9.4.0  # PIL (manejo de imágenes)
matplotlib==3.9.1  # Gráficos
pandas==2.2.2  # Procesamiento de datos
ultralytics==8.2.73  # YOLOv8
torch==2.4.0  # PyTorch para YOLOv8
scikit-learn==1.3.0  # Opcional, si usas técnicas avanzadas de ML
pyyaml==6.0  # Para manejar configuraciones YAML


"""

