import io
import json
from pathlib import Path
import PIL.Image
import tensorflow as tf
from lxml import etree
from object_detection.protos import string_int_label_map_pb2
from object_detection.utils import dataset_util


def get_all_files(directory: str, folders: list, file_format: str):
  all_files = []
  for folder in folders:
    folder_path = Path(directory) / folder
    all_files.extend(str(file.absolute()) for file in folder_path.glob(f'**/*.{file_format}'))
  return all_files

def create_folder(directory):
  try:
    directory = Path(directory) if isinstance(directory, str) else directory
    directory.mkdir(parents=True, exist_ok=True)
  except Exception as e:
    print(f'Error: Creating directory. {e}')

def dict_to_tf_example_with_json(data: dict, file_path: str, label_map_dict: dict):
  try:
    with tf.io.gfile.GFile(file_path, 'rb') as fid:
      encoded_img = fid.read()
  except Exception as e:
    raise ValueError(f'Error reading image from {file_path}: {e}')

  encoded_img_io = io.BytesIO(encoded_img)
  try:
    image = PIL.Image.open(encoded_img_io)
  except Exception as e:
    raise ValueError(f'Error opening image {file_path}: {e}')
  
  if image.format != 'JPEG':
    raise ValueError(f'Image format not JPEG: {file_path}')

  width, height = image.size
  xmin, ymin, xmax, ymax = [], [], [], []
  classes, classes_text = [], []

  try:
    for obj in data['shapes']:
      xmin.append(float(obj['points'][0][0]) / width)
      ymin.append(float(obj['points'][0][1]) / height)
      xmax.append(float(obj['points'][1][0]) / width)
      ymax.append(float(obj['points'][1][1]) / height)
      
      label = obj['label']
      if label not in label_map_dict:
        label_map_dict[label] = len(label_map_dict) + 1

      classes_text.append(label.encode('utf8'))
      classes.append(label_map_dict[label])
  except KeyError as e:
    raise ValueError(f'Missing key in data: {e}')
  except Exception as e:
    raise ValueError(f'Error processing object annotations for {data.get("imagePath", "unknown file")}: {e}')

  example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(data['imagePath'].encode('utf8')),
    'image/source_id': dataset_util.bytes_feature(data['imagePath'].encode('utf8')),
    'image/encoded': dataset_util.bytes_feature(encoded_img),
    'image/format': dataset_util.bytes_feature(image.format.lower().encode('utf8')),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
    
  return example

def dict_to_tf_example_with_xml(data: dict, file_path: str, label_map_dict: dict):
  try:
    with tf.io.gfile.GFile(file_path, 'rb') as fid:
      encoded_img = fid.read()
  except Exception as e:
    raise ValueError(f'Error reading image file: {file_path}, Error: {e}')

  encoded_img_io = io.BytesIO(encoded_img)
  image = PIL.Image.open(encoded_img_io)
  if image.format != 'JPEG':
    raise ValueError(f'Image format not JPEG: {data["filename"]}')

  width, height = image.size
  xmin, ymin, xmax, ymax = [], [], [], []
  classes, classes_text = [], []

  if 'object' in data.keys():
    for obj in data['object']:
      try:
        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)

        label = obj['name']
        if label not in label_map_dict:
          label_map_dict[label] = len(label_map_dict) + 1

        classes_text.append(label.encode('utf8'))
        classes.append(label_map_dict[label])
      except KeyError as e:
        raise ValueError(f'Missing key {e} in object data: {data["filename"]}')

  # 生成 TF Example
  example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
    'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
    'image/encoded': dataset_util.bytes_feature(encoded_img),
    'image/format': dataset_util.bytes_feature(image.format.lower().encode('utf8')),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  
  return example



def write_tf_example(save_path: str, all_files: list, label_map_dict: dict, file_format: str) -> None:
  with tf.io.TFRecordWriter(save_path) as writer:
    for idx, file in enumerate(all_files):
      if idx % 100 == 0:
        print(f'On image {idx} of {len(all_files)}')

      if file_format == 'xml':
        with tf.io.gfile.GFile(file, 'r') as fid:
          xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        tf_example = dict_to_tf_example_with_xml(data, str(Path(file).with_suffix('.jpg')), label_map_dict)

      elif file_format == 'json':
        with open(file, 'r') as fid:
          data = json.load(fid)
        tf_example = dict_to_tf_example_with_json(data, str(Path(file).with_suffix('.jpg')), label_map_dict)
      
      writer.write(tf_example.SerializeToString())

def gen_label_map(label_map_dict: dict, save_path: str) -> None:
  label_map = string_int_label_map_pb2.StringIntLabelMap()
  for name, id in label_map_dict.items():
    item = label_map.item.add()
    item.name = str(name)
    item.id = int(id)
  with tf.io.gfile.GFile(save_path, 'w') as fid:
    fid.write(str(label_map))

def generate_record(target_dir: str, data_folders: list, save_dir: str, label_map_dict={}, format='json', is_train=True):
  save_dir = Path(save_dir)
  target_dir = Path(target_dir)
  create_folder(save_dir)
  
  record_type = 'train' if is_train else 'test'
  save_record = save_dir / f'{record_type}.record'
  if is_train:
      save_label_map = save_dir / 'label_map.pbtxt'
  
  files = get_all_files(str(target_dir), data_folders, format)
  write_tf_example(str(save_record), files, label_map_dict, format)
  
  if is_train:
      gen_label_map(label_map_dict, str(save_label_map))
  
  print(f'Finish generating {record_type} set tfrecord')
  return label_map_dict if is_train else None
