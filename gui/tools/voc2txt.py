# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os



def extractVOC(vocpath, classname):
    sets = ['train', 'val', 'test']
    classes = classname   # 改成自己的类别
    abs_path = vocpath


    def convert(size, box):
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = (box[0] + box[1]) / 2.0 - 1
        y = (box[2] + box[3]) / 2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h


    def convert_annotation(imageid):
        in_file = open(os.path.join(abs_path,'Annotations/%s.xml' %imageid), encoding='UTF-8')
        out_file = open(os.path.join(abs_path,'labels/%s.txt' %imageid), 'w', encoding='UTF-8')
        # print(os.settings.join(abs_path,'labels/%s.txt' %imageid))
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            # difficult = obj.find('difficult').text
            difficult = 0
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            b1, b2, b3, b4 = b
            # 标注越界修正
            if b2 > w:
                b2 = w
            if b4 > h:
                b4 = h
            b = (b1, b2, b3, b4)
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


    for image_set in sets:
        if not os.path.exists(os.path.join(abs_path,'labels')):
            os.makedirs(os.path.join(abs_path,'labels'))
        image_ids = open(os.path.join(abs_path,'ImageSets/Main/%s.txt' % image_set)).read().strip().split()
        list_file = open(os.path.join(abs_path,'%s.txt' % image_set), 'w', encoding='UTF-8')
        for image_id in image_ids:
            list_file.write(str(os.path.join(abs_path , 'images/%s.jpg\n' % image_id)).replace("\\","/"))
            convert_annotation(image_id)
        list_file.close()
