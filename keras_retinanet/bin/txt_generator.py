"""
将图片划分为训练集和测试集，并将文件名前缀写入对应的txt文件中
"""

import os
from sklearn.model_selection import train_test_split

file_list = os.listdir("D:/jupyter_file/blood-cells/dataset-master/dataset-master/JPEGImages")
file_list = [name[:-4] for name in file_list]
train_list, test_list = train_test_split(file_list, test_size=0.5, shuffle=True, random_state=1234)
print(train_list[:5])
print(test_list[:5])
print(len(train_list), len(test_list))


def txt_generator(output_path, txt_name, list_name):
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	output_file = os.path.join(output_path, txt_name)
	f = open(output_file, 'w')
	for i in list_name:
		f.write(i)
		f.write('\n')
	f.close()


txt_generator('txtout', 'train.txt', train_list)
txt_generator('txtout', 'test.txt', test_list)
