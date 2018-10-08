import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import csv
import pandas as pd
import argparse
import copy

def get_data(input_path):
	"""
	input_path: 只需要给定到VOC所在的文件夹，如：input_path = 'F:/study_files/faster_rcnn/training_data/VOCdevkit'
	return: all_imgs, classes_count, class_mapping
	1. all_imgs: 是一个list，每一条信息是以字典形式存储包含了一张图片的所有信息。字典名称包含：图片的高度，宽度，路径和所处训练集和框。
	其中bboxes是一个list，每一条信息是以字典形式存储的包含了一个box的所有信息。有难度，类别和上下两点的坐标。如：	[{'height': 500, 
	'imageset': 'train','width': 486, 'filepath':'F:/study_files/faster_rcnn/training_data/VOCdevkit\\VOC2012\\JPEGImages\\2007_000027.jpg',
	'bboxes': [{'x2': 349, 'y1': 101, 'class': 'person', 'y2': 351, 'difficult':False, 'x1': 174}]}]
	2. classes_count: 是一个字典，其存储类别和对应的总个数，如：{'person': 2, 'horse': 1}
	3. class_mapping: 是一个字典，其存储每个类别及其对应的编号，如：{'person': 0, 'horse': 1}"""
	all_imgs = []

	classes_count = {}

	class_mapping = {}

	visualise = False
	# 只遍历VOC2007的话，可以使用下面的一行代码
	# data_paths = [os.path.join(input_path, s) for s in ['VOC2007']]

	# data_paths = [os.path.join(input_path,s) for s in ['VOC2007', 'VOC2012']]  # 这句表示遍历VOC2007和VOC2012，这取决于你的数据集
	data_path = input_path
	

	print('Parsing annotation files')


	#for data_path in data_paths:

	annot_path = os.path.join(data_path, 'Annotations')  # xml文件所在的文件夹路径
	imgs_path = os.path.join(data_path, 'JPEGImages')  # 原始图片所在的文件夹路径
	imgsets_path_train = os.path.join(data_path, 'ImageSets','Main','train.txt')
	# imgsets_path_val = os.path.join(data_path, 'ImageSets', 'Main', 'val.txt')
	imgsets_path_test = os.path.join(data_path, 'ImageSets','Main','test.txt')  # 将给定的路径和名称结合得到需要的路径

	train_files = []
	# val_files = []
	test_files = []
	try:
		with open(imgsets_path_train) as f:  # 打开文件，默认位只读模式
			"""得到训练集图片文件的名称"""
			for line in f:  # 按行读取对象类的文字
				train_files.append(line.strip() + '.jpg')  # 去除空格
	except Exception as e:
		print(e)

	# try:
	# 	with open(imgsets_path_val) as f:  # 打开文件，默认位只读模式
	# 		"""得到训练集图片文件的名称"""
	# 		for line in f:  # 按行读取对象类的文字
	# 			val_files.append(line.strip() + '.jpg')  # 去除空格
	# except Exception as e:
	# 	print(e)

	try:
		with open(imgsets_path_test) as f:
			"""得到测试集图片文件的名称"""
			for line in f:
				test_files.append(line.strip() + '.jpg')
	except Exception as e:
		if data_path[-7:] == 'VOC2012':
			# this is expected, most pascal voc distibutions dont have the test.txt file
			pass
		else:
			print(e)
	
	annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]  # 得到anot_path的所有xml文件
	idx = 0
	for annot in annots:
		"""
		开始遍历xml文件
		1. """
		try:
			idx += 1

			et = ET.parse(annot)  # 读取xml文件
			element = et.getroot()  # 得到xml的根，所有很包含的属性都可以从中得到

			element_objs = element.findall('object')  # 得到图片中框出来的对象
			element_filename = element.find('filename').text+".jpg"  # 得到图片的名称
			element_width = int(element.find('size').find('width').text)  # 得到图片的宽，由于.text是文字形式，所以使用int将其转变为数值
			element_height = int(element.find('size').find('height').text)  # 得到图片的高

			if len(element_objs) > 0:  # 首先要判断存在对象
				annotation_data = {'filepath': os.path.join(imgs_path, element_filename), 'width': element_width,
								   'height': element_height, 'bboxes': []}  # annotation_data存储这张图片的基本信息，包括路径，高和宽，框，所属数据集

				if element_filename in train_files:
					"""
					把图片所属数据集的信息加到annotation_data里面，如果属于训练集，就把它归为训练集
					"""
					annotation_data['imageset'] = 'train'
				elif element_filename in test_files:
					annotation_data['imageset'] = 'test'
				# else:
				# 	annotation_data['imageset'] = 'val'

			for element_obj in element_objs:
				"""遍历图片中所有框出来的object对象"""
				class_name = element_obj.find('name').text
				if class_name not in classes_count:
					classes_count[class_name] = 1
				else:
					classes_count[class_name] += 1  # 得到每种类别的个数

				if class_name not in class_mapping:
					class_mapping[class_name] = len(class_mapping)  # 得到每个类别对应的序号（标签）

				obj_bbox = element_obj.find('bndbox')
				x1 = int(round(float(obj_bbox.find('xmin').text)))
				y1 = int(round(float(obj_bbox.find('ymin').text)))
				x2 = int(round(float(obj_bbox.find('xmax').text)))
				y2 = int(round(float(obj_bbox.find('ymax').text)))
				difficulty = int(element_obj.find('difficult').text) == 1
				annotation_data['bboxes'].append(
					{'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
			all_imgs.append(annotation_data)  # 得到一条完整的图片信息

			if visualise:  # 是否需要读一张图片显示一张
				img = cv2.imread(annotation_data['filepath'])
				for bbox in annotation_data['bboxes']:
					"""根据框的坐标在原图上把物体框出来"""
					cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
								  'x2'], bbox['y2']), (0, 0, 255))
				cv2.imshow('img', img)
				cv2.waitKey(0)

		except Exception as e:
			print(e)
			continue
	return all_imgs, classes_count, class_mapping


# 制作csv数据集，一个包含annotation信息，另一个包含class_mapping信息
def annotation_csv(input_path, output_path):
	all_imgs, _, _, = get_data(input_path)
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	for img in all_imgs:
		for bbox in img['bboxes']:
			x1 = bbox['x1']
			y1 = bbox['y1']
			x2 = bbox['x2']
			y2 = bbox['y2']
			class_name = bbox['class']
			filepath = img['filepath']
			if class_name is not None:
				img_info = [filepath, x1, y1, x2, y2, class_name]
			else:
				img_info = [filepath, '', '', '', '', '']
			imageset = img['imageset']
			if imageset == 'train':
				output_file = os.path.join(output_path, 'annotation_train.csv')
				csvfile = open(output_file, 'w', newline='')
				writer = csv.writer(csvfile)
				writer.writerows([img_info])
				csvfile.close()
			elif imageset == 'test':
				output_file = os.path.join(output_path, 'annotation_test.csv')
				csvfile = open(output_file, 'w', newline='')
				writer = csv.writer(csvfile)
				writer.writerows([img_info])
				csvfile.close()
			else:
				output_file = os.path.join(output_path, 'annotation_val.csv')
				csvfile = open(output_file, 'w', newline='')
				writer = csv.writer(csvfile)
				writer.writerows([img_info])
				csvfile.close()
	print('annotaion_csv generate done')

# all_imgs, _, _ = get_data('D:\\jupyter_file\\blood-cells\\dataset-master\\dataset-master')
# print(all_imgs[0])
# ll_imgs: 是一个list，每一条信息是以字典形式存储包含了一张图片的所有信息。字典名称包含：图片的高度，宽度，路径和所处训练集和框。
# 其中bboxes是一个list，每一条信息是以字典形式存储的包含了一个box的所有信息。有难度，类别和上下两点的坐标。如：	[{'height': 500, 
# 'imageset': 'train','width': 486, 'filepath':'F:/study_files/faster_rcnn/training_data/VOCdevkit\\VOC2012\\JPEGImages\\2007_000027.jpg',
# 'bboxes': [{'x2': 349, 'y1': 101, 'class': 'person', 'y2': 351, 'difficult':False, 'x1': 174}]}]


def annotation_csv2(input_path, output_path):
	all_imgs, _, _, = get_data(input_path)
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	imgs = []
	for img in all_imgs:
		bboxes = img.pop("bboxes")
		for bbox in bboxes:
			img_cp = copy.deepcopy(img)
			for bbox_key, bbox_value in bbox.items():
				img_cp[bbox_key] = bbox_value
		imgs.append(img_cp)
	img_df = pd.DataFrame(imgs)
	img_df = img_df[img_df["x1"] < img_df["x2"]]
	img_df = img_df[img_df["y1"] < img_df["y2"]]
	img_df_need = img_df.loc[:, ["filepath", "x1", "y1", "x2", "y2", "class"]]
	img_df_train = img_df_need[img_df["imageset"]=="train"]
	img_df_test = img_df_need[img_df["imageset"]=="test"]
	# img_df_valid = img_df[img_df["imageset"]=="valid"]
	img_df_train.to_csv(os.path.join(output_path, "annotation_train.csv"), header=False, index=False)
	img_df_test.to_csv(os.path.join(output_path, "annotation_test.csv"), header=False, index=False)
	print('annotaion_csv generate done')


def mapping_csv(input_path, output_path):
	_, _, class_mapping = get_data(input_path)
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	output_file = os.path.join(output_path, 'mapping.csv')
	csvfile = open(output_file, 'w', newline='')
	writer = csv.writer(csvfile)
	for k, v in class_mapping.items():
		mapping_info = [k, v]
		writer.writerows([mapping_info])
	csvfile.close()
	print('mapping_csv generate done')




# def parse_args(args):
# 	parser = argparse.ArgumentParser(description='generate annotation file and classes file')
# 	subparsers = parser.add_subparsers(help='arguments for csv dataset', dest='dataset_type')
# 	subparsers.required = True

# 	img_data = subparsers.add_parser('data')
# 	img_data.add_argument('data_path', help='path to VOCdevkit')


annotation_csv2('D:/jupyter_file/blood-cells/dataset-master/dataset-master', './out')
mapping_csv('D:/jupyter_file/blood-cells/dataset-master/dataset-master', './out')