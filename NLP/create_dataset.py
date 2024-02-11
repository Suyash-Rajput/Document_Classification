import os
import cv2
import csv
from Programs.OCR import ocr
from Programs import utils
import pandas as pd
import numpy as np
from PIL import Image

def read_processed_list(store_folder):
    processed_list_pfn = os.path.join(store_folder, "processed_list.txt")
    filename_set = set()
    if not os.path.exists(processed_list_pfn):
           utils.create_text_file(processed_list_pfn)
           return filename_set
    with open(processed_list_pfn, "r") as file:
        for line in file:
            next_filename = line.strip()
            filename_set.add(next_filename)
    return filename_set

def mark_processed(product_handle, store_folder, processed_set):
    processed_list_pfn = os.path.join(store_folder, "processed_list.txt")
    if product_handle not in processed_set:
        processed_set.add(product_handle)
        with open(processed_list_pfn, "a") as file:
            file.write(product_handle + "\n")
    return processed_set      
   
def fill_dataset_given(dir_path, NLP_HOME):
    count = 0
    create_data_folder = os.path.join(NLP_HOME, "Dataset_Generator") 
    csv_file_path = os.path.join(create_data_folder, "dataset.csv")
    processed_set = read_processed_list(create_data_folder)
    if not os.path.exists(csv_file_path):
        utils.create_csv_file(csv_file_path)
    for folder_name in os.listdir(dir_path):
        data_folder = os.path.join(dir_path, folder_name)
        for file in os.listdir(data_folder):
            if not file.endswith(".jpg") and not file.endswith(".png") or file.endswith(".jpeg") :
                continue
            if file in processed_set:
                continue
            img_path = os.path.join(data_folder, file)
            text =  ocr.object_character_recognition(img_path)
            data_row = [text, folder_name]
            utils.write_row_to_csv(csv_file_path, data_row)
            processed_set = mark_processed(file, create_data_folder, processed_set)
            count+=1
        print("Folder name:", folder_name, "completed. Count:", count)

if __name__ == "__main__":
    NLP_HOME = r'E:\Newgen\NLP\Programs'
    dir_path = r'E:\Newgen\NLP\Dataset\Classification'
    fill_dataset_given(dir_path, NLP_HOME)
    