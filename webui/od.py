from modules.genRecord import generate_record
import subprocess
import shutil
from object_detection.utils import config_util, label_map_util

def export(project_name, task_name):
    try:
        process = subprocess.Popen(
            ['python', 'script/exporter_main_v2.py',
             '--trained_checkpoint_dir', f'./projects/{project_name}/Checkpoint/{task_name}',
             '--pipeline_config_path', f'./projects/{project_name}/Models/{task_name}/pipeline.config',
             '--output_directory', f'./projects/{project_name}/Models/{task_name}'
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Yield output while the command is running
        while True:
            output = process.stdout.readline()
            error_output = process.stderr.readline()
            if output:
                yield "輸出: " + output.strip() + "\n"
            if error_output:
                yield "錯誤訊息: " + error_output.strip() + "\n"

            if output == '' and error_output == '' and process.poll() is not None:
                break 

        shutil.copy(
            f'./projects/{project_name}/TFRecord/{task_name}/label_map.pbtxt',
            f'./projects/{project_name}/Models/{task_name}/label_map.pbtxt'
        )
        yield "模型轉換完成！！\n"
    except subprocess.CalledProcessError as e:
        yield "模型轉換失敗！\n" + str(e.stderr)
    except Exception as e:
        yield "模型轉換失敗！\n" + str(e)

def train(project_name, task_name, batch_size, num_steps, checkpoint_every_n, reference_model):
    try:
        # 設定模型參數
        configs = config_util.get_configs_from_pipeline_file(f'./models/{reference_model}/pipeline.config')
        label_map = label_map_util.get_label_map_dict(f'./projects/{project_name}/TFRecord/{task_name}/label_map.pbtxt')
        
        override_dict = {
            f'model.{configs["model"].WhichOneof("model")}.num_classes': len(label_map.keys()),
            'train_config.batch_size': batch_size,
            'train_config.fine_tune_checkpoint': f'./models/{reference_model}/checkpoint/ckpt-0',
            'train_config.num_steps': num_steps,
            'label_map_path': f'./projects/{project_name}/TFRecord/{task_name}/label_map.pbtxt',
            'train_input_path': f'./projects/{project_name}/TFRecord/{task_name}/train.record',
            'eval_input_path': f'./projects/{project_name}/TFRecord/{task_name}/test.record'
        }
        
        configs = config_util.merge_external_params_with_configs(configs, kwargs_dict=override_dict)
        pipeline_config = config_util.create_pipeline_proto_from_configs(configs)
        config_util.save_pipeline_config(pipeline_config, f'./projects/{project_name}/Models/{task_name}')
        yield "CONFIG 設定完成！\n"
        
        # 執行訓練程序
        process = subprocess.Popen(
            ['python', 'script/model_main_tf2.py',
             '--pipeline_config_path', f'./projects/{project_name}/Models/{task_name}/pipeline.config',
             '--model_dir', f'./projects/{project_name}/Checkpoint/{task_name}',
             '--checkpoint_every_n', str(checkpoint_every_n)
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # 實時輸出訓練進度和錯誤
        while True:
            output = process.stdout.readline()
            error_output = process.stderr.readline()
            if output:
                yield "訓練輸出: " + output.strip() + "\n"
            if error_output:
                yield "錯誤訊息: " + error_output.strip() + "\n"

            if output == '' and error_output == '' and process.poll() is not None:
                break

        if process.returncode == 0:
            yield "模型訓練完成！\n"
        else:
            yield "模型訓練可能有錯誤，請檢查！\n"
    except subprocess.CalledProcessError as e:
        yield "模型訓練失敗！\n" + str(e.stderr)
    except Exception as e:
        yield "模型訓練失敗！\n" + str(e)

def getTFRecord(project_name, dataset_format, task_name):
    try:
        # 處理訓練資料集
        for message in generate_record(
            target_dir=f'./datasets/{project_name}/train', 
            data_folders=[''],  # [''] mean selected all
            save_dir=f'./projects/{project_name}/TFRecord/{task_name}',
            format=dataset_format,
            is_train=True
        ):
            yield message

        # 處理測試資料集
        for message in generate_record(
            target_dir=f'./datasets/{project_name}/test', 
            data_folders=[''],  # [''] mean selected all
            save_dir=f'./projects/{project_name}/TFRecord/{task_name}',
            format=dataset_format,
            is_train=False
        ):
            yield message

        yield "資料轉換完成\n"
    except subprocess.CalledProcessError as e:
        yield "模型訓練失敗！\n" + e.stderr
