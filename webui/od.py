from modules.genRecord import generate_record
import subprocess
import shutil
from object_detection.utils import config_util, label_map_util

def export(project_name, task_name):
    try:
        subprocess.run(
            ['python', 'script/exporter_main_v2.py',
            '--trained_checkpoint_dir', f'./projects/{project_name}/Checkpoint/{task_name}',
            '--pipeline_config_path', f'./projects/{project_name}/Models/{task_name}/pipeline.config',
            '--output_directory', f'./projects/{project_name}/Models/{task_name}'
            ],
            # shell='cmd.exe'
        )
        shutil.copy(
            f'./projects/{project_name}/TFRecord/{task_name}/label_map.pbtxt',
            f'./projects/{project_name}/Models/{task_name}/label_map.pbtxt'
        )
        return "模型轉換完成！！\n"
    except subprocess.CalledProcessError as e:
        return "模型轉換失敗！\n" + e.stderr


def train(  project_name,
            task_name,
            batch_size, 
            num_steps, 
            checkpoint_every_n, 
            reference_model):
    try:
        ## >> CHANGE CONFIG
        configs = config_util.get_configs_from_pipeline_file(f'./models/{reference_model}/pipeline.config')
        label_map = label_map_util.get_label_map_dict(f'./projects/{project_name}/TFRecord/{task_name}/label_map.pbtxt')
        override_dict = {
            f'model.{configs["model"].WhichOneof("model")}.num_classes': len(label_map.keys()),
            'train_config.batch_size': batch_size,
            'train_config.fine_tune_checkpoint': f'./models/{reference_model}/checkpoint/ckpt-0',
            'train_config.num_steps': num_steps,
            # legacy update
            'label_map_path': f'./projects/{project_name}/TFRecord/{task_name}/label_map.pbtxt',
            'train_input_path': f'./projects/{project_name}/TFRecord/{task_name}/train.record',
            'eval_input_path': f'./projects/{project_name}/TFRecord/{task_name}/test.record'
        }
        configs = config_util.merge_external_params_with_configs(configs, kwargs_dict=override_dict,)
        pipeline_config = config_util.create_pipeline_proto_from_configs(configs)
        config_util.save_pipeline_config(pipeline_config, f'./projects/{project_name}/Models/{task_name}')
        print('CHANGE CONFIG DONE!!')
        subprocess.run(
            ['python', 'script/model_main_tf2.py',
            '--pipeline_config_path', f'./projects/{project_name}/Models/{task_name}/pipeline.config',
            '--model_dir', f'./projects/{project_name}/Checkpoint/{task_name}',
            '--checkpoint_every_n',f"{checkpoint_every_n}"
            ],
            # shell='cmd.exe'
        )
        return "模型訓練完成！\n"
    except subprocess.CalledProcessError as e:
        return "模型訓練失敗！\n" + e.stderr

def getTFRecord(project_name, dataset_format, task_name):
    try:
        label_map_dict = generate_record(
            target_dir=f'./datasets/{project_name}/train', 
            data_folders=[''],  # [''] mean selected all
            save_dir=f'./projects/{project_name}/TFRecord/{task_name}',
            format=dataset_format,
            is_train=True
        )
        generate_record(
            target_dir=f'./datasets/{project_name}/test', 
            data_folders=[''],  # [''] mean selected all
            save_dir=f'./projects/{project_name}/TFRecord/{task_name}',
            format=dataset_format,
            is_train=False
        )
        return "資料轉換完成\n"
    except subprocess.CalledProcessError as e:
        return "模型訓練失敗！\n" + e.stderr
