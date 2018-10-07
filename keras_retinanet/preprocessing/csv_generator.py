"""
Copyright 2017-2018 yhenon (https://github.com/yhenon/)
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from .generator import Generator
from ..utils.image import read_image_bgr

import numpy as np
from PIL import Image
from six import raise_from

import csv
import sys
import os.path


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)  # 就是将输入代入某个函数，得到返回的结果。如果出错，返回提示信息而已
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_classes(csv_reader):
    """ Parse the classes file given by csv_reader.
    返回的是一个字典，形式为{'chair': 0, 'car': 1, 'horse': 2}，即{'class_name': class_id}
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def _read_annotations(csv_reader, classes):
    """ Read annotations from the csv_reader.
    input: 
    1. csv_reader: 是一个读取器，返回的是多行内容
    2. classes: 是一个字典，是由_read_classes函数得到的

    得到的result为一个字典，其中包含了annotation中的信息
        形如：{'D:/jupyter_file/keras_frcnn_git/VOCdevkit\\VOC2007\\JPEGImages\\000005.jpg': 
        [{'x1': 263, 'x2': 324, 'y1': 211, 'y2': 339, 'class': 'chair'}, 
        {'x1': 165, 'x2': 253, 'y1': 264, 'y2': 372, 'class': 'chair'}, 
        {'x1': 5, 'x2': 67, 'y1': 244, 'y2': 374, 'class': 'chair'}, 
        {'x1': 241, 'x2': 295, 'y1': 194, 'y2': 299, 'class': 'chair'}, 
        {'x1': 277, 'x2': 312, 'y1': 186, 'y2': 220, 'class': 'chair'}], 
        'D:/jupyter_file/keras_frcnn_git/VOCdevkit\\VOC2007\\JPEGImages\\000007.jpg': 
        [{'x1': 141, 'x2': 500, 'y1': 50, 'y2': 330, 'class': 'car'}]}
    """
    result = {}
    for line, row in enumerate(csv_reader):
        # 从读取器中把内容读取出来
        line += 1

        try:
            img_file, x1, y1, x2, y2, class_name = row[:6]  # 将读取的内容关联到对应的名称
        except ValueError:
            raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
            continue

        # 因为上面返回的坐标信息是以str表示的，所以这里需要将其转变为数值形式
        x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

        # Check that the bounding box is valid.
        # 检查bounding-box是否正常符合逻辑
        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

        # check if the current class name is correctly present
        # 判断类别是否在给定的类别字典中
        if class_name not in classes:
            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

        result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
    return result


def _open_for_csv(path):
    """ Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    针对不同版本的python分别给出了解决办法，目的是打开csv文件
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


class CSVGenerator(Generator):
    """ Generate data for a custom CSV dataset.

    See https://github.com/fizyr/keras-retinanet#csv-datasets for more information.
    """

    def __init__(
        self,
        csv_data_file,  # annotation file
        csv_class_file,  # class file
        base_dir=None,
        **kwargs
    ):
        """ Initialize a CSV data generator.

        Args
            csv_data_file: Path to the CSV annotations file.
            csv_class_file: Path to the CSV classes file.
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
        """
        self.image_names = []  # 用于存放文件名
        self.image_data  = {}  # 用于存放annotation信息
        self.base_dir    = base_dir  # csv文件所在的路径的主目录

        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            """
            如果没有给定标注文件和类别映射文件所在的路径的主路径，可以使用下面的os.path.dirname
            得到主路径
            """
            self.base_dir = os.path.dirname(csv_data_file)

        # parse the provided class file
        try:
            with _open_for_csv(csv_class_file) as file:
                self.classes = _read_classes(csv.reader(file, delimiter=','))  # 此处得到的classes是个字典，形如{class_name: class_id}
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(csv_class_file, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key  # 从classes字典中得到类别名称添加到label中，并将其键值对翻转，形如{0: 'car', 1: 'horse'}

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with _open_for_csv(csv_data_file) as file:
                # 打开csv文件，这个文件是图片信息annotation所在的文件
                self.image_data = _read_annotations(csv.reader(file, delimiter=','), self.classes) # 得到的是一个包含annotation信息的字典
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_data_file, e)), None)
        self.image_names = list(self.image_data.keys())  # 得到图片名称构成的列表

        super(CSVGenerator, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        图片的数量
        """
        return len(self.image_names)

    def num_classes(self):
        """ Number of classes in the dataset.
        有多少类别
        """
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        """ Map name to label.
        名称和标签编号对应的映射，作用是获取名称对应的标签编号
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        标签编号和名称的映射，作用是获取标签编号对应的名称
        """
        return self.labels[label]

    def image_path(self, image_index):
        """ Returns the image path for image_index.
        # 作用是获取指定图片的路径，比如获取列表中第2张图片的路径，依据图片索引便可以得到它的路径
        """
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        获取指定图片的宽高比
        """
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        # 依据索引加载图片，并且加载进来的是BGR格式的图片数组
        """
        return read_image_bgr(self.image_path(image_index))

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        获取指定图片中的所有bounding-box的annotation信息
        最终返回的是bounding-box信息，形如
        [[263. 211. 324. 339.   0.]
        [165. 264. 253. 372.   0.]
        [  5. 244.  67. 374.   0.]
        [241. 194. 295. 299.   0.]
        [277. 186. 312. 220.   0.]]
        其中前四列分别为x1, y1, x2, y2，第五列为class类别
        """
        path   = self.image_names[image_index]  # 图片路径
        annots = self.image_data[path]  # 依据图片路径对应的键，得到其值，即该图片对应的annotation信息，每个图片对应的annotation信息为一个列表
        boxes  = np.zeros((len(annots), 5))  # 每一个bounding-box对应的annotation信息由x1, x2, y1, y2, class五部分组成，所以这里建立一个5列数组待用

        for idx, annot in enumerate(annots):
            class_name = annot['class']
            boxes[idx, 0] = float(annot['x1'])
            boxes[idx, 1] = float(annot['y1'])
            boxes[idx, 2] = float(annot['x2'])
            boxes[idx, 3] = float(annot['y2'])
            boxes[idx, 4] = self.name_to_label(class_name)

        return boxes
