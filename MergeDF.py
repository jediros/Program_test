# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:40:31 2024

@author: Jedi Rosero
"""
import pandas as pd

class MergeDF:
    def __init__(self, bbox_csv_path, contour_areas_csv_path, output_csv_path):
        self.bbox_csv_path = bbox_csv_path
        self.contour_areas_csv_path = contour_areas_csv_path
        self.output_csv_path = output_csv_path

    def merge_csv_files(self):
        # Read CSV files
        bbox_df = pd.read_csv(self.bbox_csv_path)
        contour_areas_df = pd.read_csv(self.contour_areas_csv_path)

        # Merge based on row index
        merged_df = pd.concat([bbox_df, contour_areas_df.iloc[:, -1]], axis=1)
        
        # Display or use the merged dataframe as needed
        print("Merged DataFrame:")
        print(merged_df)

        # Save the merged DataFrame to CSV
        merged_df.to_csv(self.output_csv_path, index=False)
        
        print(f"Merged DataFrame saved to {self.output_csv_path}")

        return merged_df
