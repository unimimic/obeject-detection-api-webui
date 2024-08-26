import os
import json
import requests
import tarfile
import gradio as gr

models = [
    "faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8",
    "faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8"
]

def download_model(model_name):
    models_path = "models"
    model_url = f"http://download.tensorflow.org/models/object_detection/tf2/20200711/{model_name}.tar.gz"
    temp_file_path = os.path.join(models_path, f"{model_name}.tar.gz")

    if not os.path.exists(models_path):
        os.makedirs(models_path)
    
    # 下載模型
    response = requests.get(model_url)
    
    if response.status_code == 200:
        with open(temp_file_path, "wb") as f:
            f.write(response.content)
        
        # 解壓縮模型
        try:
            with tarfile.open(temp_file_path, "r:gz") as tar:
                tar.extractall(path=models_path)
            os.remove(temp_file_path)  # 刪除暫存的 tar.gz 文件
            return f"模型 '{model_name}' 下載並解壓成功。"
        except Exception as e:
            return f"解壓縮模型 '{model_name}' 時發生錯誤: {str(e)}"
    else:
        return f"模型 '{model_name}' 下載失敗，狀態碼: {response.status_code}。"

def get_models():
    models_path = "models"
    models = []

    if os.path.exists(models_path) and os.path.isdir(models_path):
        for model_name in os.listdir(models_path):
            models.append(model_name)

    return models

def save_project_settings(project_name, dataset_format, training_classes, batch_size, num_steps, checkpoint_every_n, model, task_name):
    project_path = os.path.join("projects", project_name)
    settings_file = os.path.join(project_path, "setting.json")
    
    settings = {
        "required": {
            "Model": model,
            "format": dataset_format,
            "TFRecord": task_name,
            "labels": training_classes.split(",") if training_classes else []
        },
        "optional": {
            "batch_size": batch_size,
            "num_steps": num_steps,
            "checkpoint_every_n": checkpoint_every_n
        }
    }
    
    with open(settings_file, "w") as f:
        json.dump(settings, f, indent=2)
    
    return f"專案 '{project_name}' 的設定已成功存取。"

def update_fields(project_name):
    settings = load_project_settings(project_name)
    
    if settings:
        return (
            settings["required"]["format"],
            ','.join(settings["required"]["labels"]),
            settings["optional"]["batch_size"],
            settings["optional"]["num_steps"],
            settings["optional"]["checkpoint_every_n"],
            settings["required"]["Model"],
            settings["required"]["TFRecord"]
        )
    return ("", "", 1, 1, 1, "", "")

def load_project_settings(project_name):
    project_path = os.path.join("projects", project_name)
    settings_file = os.path.join(project_path, "setting.json")

    if not os.path.exists(settings_file):
        return None
    
    with open(settings_file, "r") as f:
        settings = json.load(f)
    
    return settings

def get_project_names():
    project_base_path = "projects"
    project_names = []

    if not os.path.exists(project_base_path):
        os.makedirs(project_base_path)

    for folder_name in os.listdir(project_base_path):
        folder_path = os.path.join(project_base_path, folder_name)
        if os.path.isdir(folder_path) and "setting.json" in os.listdir(folder_path):
            project_names.append(folder_name)
    
    return project_names

def create_project_directory(project_name):    
    if not project_name:
        return "名稱不能為空"
    
    project_path = os.path.join("projects", project_name)
    project_dataset_path = os.path.join("datasets", project_name)

    try:
        if not os.path.exists(project_path):
            os.makedirs(project_path)
            os.makedirs(project_dataset_path)
            # 建立子資料夾
            os.makedirs(os.path.join(project_path, "Checkpoint"))
            os.makedirs(os.path.join(project_path, "Models"))
            os.makedirs(os.path.join(project_path, "TFRecord"))
            os.makedirs(os.path.join(project_dataset_path, "train"))
            os.makedirs(os.path.join(project_dataset_path, "test"))
            
            # 建立setting.json
            settings = {
                "required": {
                    "Model": "",
                    "format": "json",
                    "TFRecord": "",
                    "labels": []
                },
                "optional": {
                    "batch_size": 3,
                    "num_steps": 90000,
                    "checkpoint_every_n": 10000,
                    "fine_tune_checkpoint_type": "detection",
                    "use_bfloat16": False
                }
            }
            
            with open(os.path.join(project_path, "setting.json"), "w") as f:
                json.dump(settings, f, indent=2)
            
            return (
                gr.update(value= f"專案資料夾 {project_name} 及其子資料夾和設定檔已成功建立。"),
                gr.update(choices=get_project_names()),  # 更新專案名稱下拉選單
            )
        else:
            return (
                gr.update(value=f"專案資料夾 {project_name} 已存在。"),
                gr.update(choices=get_project_names()),  # 更新專案名稱下拉選單
            )
    except Exception as e:
        return (
            gr.update(value=f"建立專案資料夾時發生錯誤: {str(e)}"),
            gr.update(choices=get_project_names()), 
        )


def process_string_list(input_text):
    # 將輸入字串拆分為列表
    string_list = input_text.split(',')
    # 在這裡處理字串列表
    processed_list = [s.strip() for s in string_list]
    return processed_list