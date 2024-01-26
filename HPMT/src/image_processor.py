import xml.dom.minidom
import os
import numpy as np
from collections import Counter
import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms
import os
from collections import Counter
from PIL import Image
import shutil
import pickle
TARGET_IMG_SIZE = 224
img_to_tensor = transforms.ToTensor()

def make_model():
    model=models.vgg16(pretrained=True)   # 其实就是定位到第28层，对照着上面的key看就可以理解
    model=model.eval()    # 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
    model.cuda()    # 将模型从CPU发送到GPU,如果没有GPU则删除该行
    return model

# def make_model():
#     model=models.resnet50(pretrained=True)   # 其实就是定位到第28层，对照着上面的key看就可以理解
#     model=model.eval()    # 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
#     model.cuda()    # 将模型从CPU发送到GPU,如果没有GPU则删除该行
#     return model

#特征提取
def extract_feature(model,imgpath):
    model.eval()      # 必须要有，不然会影响特征提取结果
    try:
        img=Image.open(imgpath)       # 读取图片
    except:
        return 'None'

    img=img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    tensor=img_to_tensor(img) # 将图片转化成tensor
    tensor=tensor.cuda()    # 如果只是在cpu上跑的话要将这行去掉
    if tensor.shape[0] == 3:
        result=model(Variable(tensor.unsqueeze(0)))
        result_npy=result.data.cpu().numpy()  # 保存的时候一定要记得转成cpu形式的，不然可能会出错

        return result_npy[0]  # 返回的矩阵shape是[1, 512, 14, 14]，这么做是为了让shape变回[512, 14,14]
    else:
      return 'None'
# model=make_model()
# number = 0
# file_path = '/home/ltf/code/data/multi-modal/data_zero_newadd'
# all_feature = []
# name_list = os.listdir(file_path)
# pdfFiles = [f for f in name_list]
# for pdf in pdfFiles:
#     number += 1
#     print(number)
#     pdfPath = os.path.join(file_path,pdf)
# #     all_feature.append(len(os.listdir(pdfPath)))
# # print(Counter(all_feature))
#     for image in os.listdir(pdfPath):
#         new_path = os.path.join(pdfPath, image)
#         image_feature = extract_feature(model, new_path)
#         if image_feature == 'None':
#             continue
#         else:
#             all_feature.append(image_feature.tolist())
#     if len(all_feature) == 0:
#         print(pdfPath)
#         os.rmdir(pdfPath)
# model=make_model()
# number = 0
# json_path = '/home/ltf/code/data/data_two_newadd_json/data_one_train.json'
# image_path = '/home/ltf/code/data/multi-modal/data_zero_newadd'
# name_list = os.listdir(image_path)
# pdfFiles = [f for f in name_list]
# n = 0
# for pdf in pdfFiles:
#     n += 1
#     print(n)
    # pdfPath = os.path.join(image_path,pdf)
    # all_feature = []
    # for image in os.listdir(pdfPath):
    #     new_path = os.path.join(pdfPath, image)
    #     image_feature = extract_feature(model, new_path)
    #     if image_feature == 'None':
    #         continue
    #     else:
    #         all_feature.append(image_feature.tolist())
    # if len(all_feature)==0:
    #     print(pdf)
    #     if len(os.listdir(pdfPath)) == 0:
    #         os.rmdir(pdfPath)
    #     else:
    #         for image in os.listdir(pdfPath):
    #             new_path = os.path.join(pdfPath, image)
    #             os.remove(new_path)
    #         os.rmdir(pdfPath)
# model=make_model()
# number = 0
# image_path = '/home/ltf/code/data/multi-modal/MAAPD-image'      #12761
# image_path1 = '/home/ltf/code/data/multi-modal/MAAPD1-image'    #4459         4
# image_path2 = '/home/ltf/code/data/multi-modal/ltf-txt-pdf'    #34434  17217   33484 16698
# all_liuxing = []
# name_list = os.listdir(image_path2)
# pdfFiles = [f for f in name_list if f.endswith('.xml')]
# print(len(pdfFiles))
# n = 0
# for pdf in pdfFiles:
#     n += 1
#     # print(n)
#     pdfPath = os.path.join(image_path1,pdf[:-4])
#     pdfPath1 = os.path.join(image_path, pdf[:-4])
#     if os.path.exists(pdfPath) or os.path.exists(pdfPath1):
#         continue
#     else:
#         print(pdf)
        # os.remove(os.path.join(image_path2,pdf))
        # os.remove(os.path.join(image_path2, pdf[:-4] + '.txt'))
#     all_feature = []
#     for image in os.listdir(pdfPath):
#         new_path = os.path.join(pdfPath,image)
#         image_feature = extract_feature(model, new_path)
#         if image_feature == 'None':
#             continue
#         else:
#             all_feature.append(image_feature.tolist())
#     if len(all_feature) == 0 and os.path.exists(os.path.join(image_path2,pdf+'.txt')):
#         os.remove(os.path.join(image_path2,pdf+'.txt'))
#         os.remove(os.path.join(image_path2,pdf+'.xml'))
    #*******************************************************************************************************************

# model=make_model()
# number = 0
# json_path = '/home/ltf/code/data/multi-modal/data_all1/data_all_train.json'
# image_path = '/home/ltf/code/data/multi-modal/MAAPD-image'
# image_path1 = '/home/ltf/code/data/multi-modal/MAAPD-image1'
# all_liuxing = []
# for line in open(json_path):
#     number += 1
#     print(number)
#     text = line.split('\t')
#     title = eval(text[1])['title'].strip()
#
#     #最后计算流形特征  用特征
#     all_feature = []
#     if os.path.exists(os.path.join(image_path,title)):
#         for image in os.listdir(os.path.join(image_path,title)):
#             new_path = os.path.join(os.path.join(image_path,title),image)
#             image_feature = extract_feature(model, new_path)
#             if image_feature == 'None':
#                 continue
#             else:
#                 all_feature.append(image_feature.tolist())
#     else:
#         for image in os.listdir(os.path.join(image_path1,title)):
#             new_path = os.path.join(os.path.join(image_path1,title),image)
#             image_feature = extract_feature(model, new_path)
#             if image_feature == 'None':
#                 continue
#             else:
#                 all_feature.append(image_feature.tolist())
#     #*******************************************************************************************************************
#     all_liuxing.append(all_feature)
#
# list_file = open('/home/ltf/code/data/multi-modal/data_all1/train_vgg.img','wb')
# pickle.dump(all_liuxing,list_file)
# list_file.close()

# model=make_model()
# number = 0
# json_path = '/home/ltf/code/data/multi-modal/MAAPD-final/data_all_dev.json'
# image_path = '/home/ltf/code/data/multi-modal/MAAPD-image'
# image_path1 = '/home/ltf/code/data/multi-modal/MAAPD1-image'
# all_liuxing = []
# for line in open(json_path):
#     number += 1
#     print(number)
#     text = line.split('\t')
#     title = eval(text[1])['title'].strip()
#
#     #最后计算流形特征  用特征
#     all_feature = []
#     if os.path.exists(os.path.join(image_path,title)):
#         for image in os.listdir(os.path.join(image_path,title)):
#             new_path = os.path.join(os.path.join(image_path,title),image)
#             image_feature = extract_feature(model, new_path)
#             if image_feature == 'None':
#                 continue
#             else:
#                 all_feature.append(image_feature.tolist())
#     else:
#         for image in os.listdir(os.path.join(image_path1,title)):
#             new_path = os.path.join(os.path.join(image_path1,title),image)
#             image_feature = extract_feature(model, new_path)
#             if image_feature == 'None':
#                 continue
#             else:
#                 all_feature.append(image_feature.tolist())
#     #*******************************************************************************************************************
#     all_liuxing.append(all_feature)
#
# list_file = open( '/home/ltf/code/data/multi-modal/MAAPD-final/dev_vgg.img','wb')
# pickle.dump(all_liuxing,list_file)
# list_file.close()

import os
import random
import numpy as np
f_train1 = open('/home/ltf/code/data/multi-modal/MAAPD-final/data_all_train.tsv','w',encoding='utf-8')
for line in open('/home/ltf/code/data/multi-modal/MAAPD-final/data_all_train.json'):
    text  = line.split('\t')
    label = text[0]
    b = ''
    number = 0
    for key,value in eval(text[1]).items():
        if key != 'title':
            abstract = value.strip()
            b += abstract
            b += ' '
    f_train1.writelines(label + '\t' + b.strip())
    f_train1.write('\n')
f_train1.close()

# import cv2 as cv#导入OpenCV模块
# import numpy as np
# import os
# import shutil
# file_path = '/home/ltf/code/data_one_newadd'
# name_list = os.listdir(file_path)
# pdfFiles = [f for f in name_list]
# print(len(pdfFiles))
# n = 0
# num = 0
# all_length = []
# for pdf in pdfFiles:
#     n += 1
#     print(n)
#     pdfPath = os.path.join(file_path,pdf)

#     for image in os.listdir(pdfPath):
#         newpath = os.path.join(pdfPath,image)
#     #     size = os.path.getsize(newpath)
#     #     if float(size)/1024 <= 10:
#     #         n += 1
#     #         print(n)
#             # os.remove(pdfPath)
#     #     ******************************************
#     #     for image in os.listdir(pdfPath):
#     #         newpath = os.path.join(pdfPath,image)
#     #         newpath1 = os.path.join(file_path1,image)
#     #         if os.path.exists(newpath1):
#     #             continue
#     #         else:
#     #             shutil.move(newpath,file_path2)
#     #     ******************************
#     #     if int(pdf[:-4]) <= 325778:
#     #         shutil.move(pdfPath,file_path1)
#     #     ***************************************
#     #     ***************************************
#         src = cv.imread(newpath)
#         if not type(src) is np.ndarray:
#             num += 1
#             print(num)
#             os.remove(newpath)
#         else:
#             if (src.shape[0] // src.shape[1]) > 10 or (src.shape[1] // src.shape[0]) > 10:
#                 os.remove(newpath)
    # #     *********************************************

# import os
# json_path = '/home/ltf/code/data/data_one_newadd_json/data_one_test.json'
# f_train = open(r'/home/ltf/code/data/data_one_newadd_json/data_one_test1.json', 'w')
# number = 0
# image_path = '/home/ltf/code/data_one_newadd'
# for line in open(json_path):
#     text = line.split('\t')
#     title = eval(text[1])['title'].strip()
#
#     if os.path.exists(os.path.join(image_path,title)):
#         f_train.writelines(line)
#     else:
#         number += 1
#         print(number)
#         # print(title)
# f_train.close()

# filedir = '/home/ltf/code/data/multi-modal/data_all/data_all.json'
# filedir1 = '/home/ltf/code/data/multi-modal/data_zero_newadd'
# filedir2 = '/home/ltf/code/data/multi-modal/data_two_newadd1'
# # f = open('G:/Arxiv/data_all/data_two_all1.json','w')
# n = 0
# for line in open(filedir):
#     text = line.split('\t')
#     title = eval(text[1])['title']
#     if os.path.exists(os.path.join(filedir1,title)) or os.path.exists(os.path.join(filedir2,title)):
#         n += 1
#         print(n)
#         # f.writelines(line)
#     else:
#         continue

# topic_num_map = {"cs.lg": 0, "cs.ai": 1, "stat.ml": 2,"cs.ni": 3, "cs.cv": 4, "cs.cl": 5, "cs.cr": 6,"cs.cy": 7,"cs.hc": 8, "cs.ro": 9,}
# def get_binary_label(topics):
#     """
#     Get a 90-digit binary encoded label corresponding to the given topic list
#     :param topics: Set of topics to which an article belongs
#     :return: 90-digit binary one-hot encoded label
#     """
#     category_label = [0 for x in range(len(topic_num_map))]
#     # for topic in topics:
#     if topics[0].strip() in topic_num_map:
#         category_label[topic_num_map[topics[0].strip()]] = 1
#
#     if sum(category_label) > 0:
#         return ''.join(map(str, category_label))
#     else:
#         print("Label", topics)
#         return None
#
# import os
# import xml.dom.minidom
# import json
# import shutil
# file_path = '/home/ltf/code/data/multi-modal/ltf-txt-pdf'
# name_list = os.listdir(file_path)
# xmlFiles = [f for f in name_list if f.endswith(".xml")]
# for number,xml_file in enumerate(xmlFiles):
#     paper_dict = {}
#     txtPath = os.path.join(file_path,xml_file[:-4]+'.txt')
#     #******title
#     human_label = open(txtPath, 'r').read()
#     topics = list(filter(None, human_label.split(' ')))
#     label = get_binary_label(topics)
#
#     paper_dict['title'] = xml_file[:-4]
#     try:
#         dom = xml.dom.minidom.parse(os.path.join(file_path,xml_file))
#         root = dom.documentElement
#
#         # ******abstract
#         Abstract = root.getElementsByTagName('abstract')
#         if not len(Abstract[0].getElementsByTagName('p')) == 0:
#             # paper_dict['abstract'] = Abstract[0].getElementsByTagName('p')[0].childNodes[0].data
#             new_abstract = Abstract[0].getElementsByTagName('p')
#             all_abstract = ''
#             for ab in new_abstract:
#                 element = ''
#                 for pp_element in ab.childNodes:
#                     if pp_element.nodeType == 3:
#                         element += pp_element.nodeValue
#                 all_abstract += element
#             paper_dict['abstract'] = all_abstract
#
#         # **********************body
#         Body = root.getElementsByTagName('body')  # 如果没有body，则跳过
#         if len(Body) == 0:
#             continue
#         else:
#             n = 0
#             Div = Body[0].getElementsByTagName('div')
#             for div in Div:
#                 head = div.getElementsByTagName('head')
#                 if not len(head) == 0:
#                     key = head[0].childNodes[0].data
#                 else:
#                     key = n  # 循环每一个div标签
#                 div_content = div.getElementsByTagName('p')
#                 all_element = ''
#                 if len(div_content) == 0:
#                     continue
#                 else:
#                     for p in div_content:
#                         # p_element = p.childNodes
#                         element = ''
#                         for pp_element in p.childNodes:
#                             if pp_element.nodeType == 3:
#                                 element += pp_element.nodeValue
#                         all_element += element
#                     paper_dict[key] = all_element
#                 n += 1
#         json_str = json.dumps(paper_dict)
#         with open('/home/ltf/code/data/multi-modal/MAAPD-final/data_all.json', 'a') as json_file:
#             json_file.write(label + '\t' + json_str + '\n')
#     except:
#         continue



# import os
# import random
# import numpy as np
# from collections import Counter
# all_section = []
# for line in open('/home/ltf/code/data/multi-modal/MAAPD-final/data_all_final.json'):
#     text  = line.split('\t')
#     label = text[0]
#     all_section.append(label)
# print(Counter(all_section))
import re
def _clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

# import os
# import random
# import json
# import numpy as np
# from collections import Counter
# from nltk.tokenize import sent_tokenize
# f_test = open('/home/ltf/code/data/multi-modal/MAAPD-final/data_all.json', 'r')
# all_length1 = []
# all_length = []
# n = 0
# num = 0
# for line in f_test:
#     text  = line.split('\t')
#     label = text[0]
#     # num += 1
#     # print(num)
#     paper_dict = {}
#     paper_dict['title'] = eval(text[1])['title']
#     for key,value in eval(text[1]).items():
#
#     # if len(eval(text[1])) <= 1:
#     #     num += 1
#     #     print(num)
#         # continue
#     # if 'abstract' in eval(text[1]).keys():
#     #     num += 1
#     #     print(num)
#     #     f_train.writelines(line)
# #     else:
# #         f_train.writelines(line)
# # f_train.close()
#         if key == 'title':
#             continue
#         else:
#             paper_dict[key] = _clean_str(value)
#     json_str = json.dumps(paper_dict)
#     with open('/home/ltf/code/data/multi-modal/MAAPD-final/data_all_final.json','a') as json_file:
#         json_file.write(label + '\t' + json_str + '\n')


# import os
# import random
# from collections import Counter
# filedir = '/home/ltf/code/data/multi-modal/MAAPD-final/data_all_final.json'
# f_train = open('/home/ltf/code/data/multi-modal/MAAPD-final/data_all_train.json', 'w',encoding = 'utf-8')
# f_test = open('/home/ltf/code/data/multi-modal/MAAPD-final/data_all_test.json', 'w',encoding = 'utf-8')
# f_val = open('/home/ltf/code/data/multi-modal/MAAPD-final/data_all_dev.json', 'w',encoding = 'utf-8')
# # #先把7类分成7个列表
# zero = []
# one = []
# two = []
# three = []
# four = []
# five = []
# six = []
# seven = []
# eight = []
# nine = []
# num = 0
#
# for line in open(filedir,encoding='utf-8'):
#     text = line.split('\t')
#     label = text[0]
#     if label == '1000000000':
#         zero.append(num)
#     elif label == '0100000000':
#         one.append(num)
#     elif label == '0010000000':
#         two.append(num)
#     elif label == '0001000000':
#         three.append(num)
#     elif label == '0000100000':
#         four.append(num)
#     elif label == '0000010000':
#         five.append(num)
#     elif label == '0000001000':
#         six.append(num)
#     elif label == '0000000100':
#         seven.append(num)
#     elif label == '0000000010':
#         eight.append(num)
#     elif label == '0000000001':
#         nine.append(num)
#     num += 1
# zero_test_indice = random.sample(zero, 341)              #测试集索引
# zero_a1 = [i for i in zero if i not in zero_test_indice]
# zero_val_indice = random.sample(zero_a1, 341)              #验证集索引
# zero_train_indice = [i for i in zero_a1 if i not in zero_val_indice]       #训练集索引
#
# one_test_indice = random.sample(one, 244)              #测试集索引
# one_a1 = [i for i in one if i not in one_test_indice]
# one_val_indice = random.sample(one_a1, 244)              #验证集索引
# one_train_indice = [i for i in one_a1 if i not in one_val_indice]       #训练集索引
#
# two_test_indice = random.sample(two, 155)              #测试集索引   875
# two_a1 = [i for i in two if i not in two_test_indice]
# two_val_indice = random.sample(two_a1, 155)              #验证集索引
# two_train_indice = [i for i in two_a1 if i not in two_val_indice]       #训练集索引
#
# three_test_indice = random.sample(three, 175)              #测试集索引  89222-63511=25711
# three_a1 = [i for i in three if i not in three_test_indice]
# three_val_indice = random.sample(three_a1, 175)              #验证集索引
# three_train_indice = [i for i in three_a1 if i not in three_val_indice]       #训练集索引
#
# four_test_indice = random.sample(four, 613)              #测试集索引
# four_a1 = [i for i in four if i not in four_test_indice]
# four_val_indice = random.sample(four_a1, 613)              #验证集索引
# four_train_indice = [i for i in four_a1 if i not in four_val_indice]       #训练集索引
#
# five_test_indice = random.sample(five, 385)              #测试集索引
# five_a1 = [i for i in five if i not in five_test_indice]
# five_val_indice = random.sample(five_a1, 385)              #验证集索引
# five_train_indice = [i for i in five_a1 if i not in five_val_indice]       #训练集索引
#
# six_test_indice = random.sample(six, 277)              #测试集索引
# six_a1 = [i for i in six if i not in six_test_indice]
# six_val_indice = random.sample(six_a1, 277)              #验证集索引
# six_train_indice = [i for i in six_a1 if i not in six_val_indice]       #训练集索引
#
# seven_test_indice = random.sample(seven, 162)              #测试集索引
# seven_a1 = [i for i in seven if i not in seven_test_indice]
# seven_val_indice = random.sample(seven_a1, 162)              #验证集索引
# seven_train_indice = [i for i in seven_a1 if i not in seven_val_indice]       #训练集索引
#
# eight_test_indice = random.sample(eight, 208)              #测试集索引
# eight_a1 = [i for i in eight if i not in eight_test_indice]
# eight_val_indice = random.sample(eight_a1, 208)              #验证集索引
# eight_train_indice = [i for i in eight_a1 if i not in eight_val_indice]       #训练集索引
#
# nine_test_indice = random.sample(nine, 440)              #测试集索引
# nine_a1 = [i for i in nine if i not in nine_test_indice]
# nine_val_indice = random.sample(nine_a1, 440)              #验证集索引
# nine_train_indice = [i for i in nine_a1 if i not in nine_val_indice]       #训练集索引
#
# number = 0
# for line in open(filedir,encoding='utf-8'):
#     if number in zero_train_indice + one_train_indice + two_train_indice + three_train_indice + four_train_indice + five_train_indice + six_train_indice + seven_train_indice + eight_train_indice + nine_train_indice:
#         f_train.writelines(line)
#     elif number in zero_test_indice + one_test_indice + two_test_indice + three_test_indice + four_test_indice + five_test_indice + six_test_indice + seven_test_indice + eight_test_indice + nine_test_indice:
#         f_test.writelines(line)
#     elif number in zero_val_indice + one_val_indice + two_val_indice + three_val_indice + four_val_indice + five_val_indice + six_val_indice + seven_val_indice + eight_val_indice + nine_val_indice:
#         f_val.writelines(line)
#     else:
#         continue
#     number += 1
# f_train.close()
# f_test.close()
# f_val.close()

