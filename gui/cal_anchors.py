import os
import numpy as np
import xml.etree.cElementTree as et
from kmeans import kmeans, avg_iou

def cal_main(vocpath, classesname):
    classnum = len(classesname)
    STR_s = """
# Parameters
nc: {}  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
anchors:
  - [{},{}, {},{}, {},{}]  
  - [{},{}, {},{}, {},{}]  
  - [{},{}, {},{}, {},{}]  

# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, C3, [1024, False]],  # 9
  ]

# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
]
"""

    STR_yaml = """train: {}
val: {}
# numbel of classes
nc: {}
# class names
names: {}
"""

    FILE_ROOT = vocpath     # 根路径
    ANNOTATION_ROOT = "Annotations"  # 数据集标签文件夹路径
    ANNOTATION_PATH = os.path.join(FILE_ROOT , ANNOTATION_ROOT)

    ANCHORS_TXT_PATH = os.path.join(FILE_ROOT,"anchors.txt")

    CLUSTERS = 9
    CLASS_NAMES = classesname

    def load_data(anno_dir, class_names):
        xml_names = os.listdir(anno_dir)
        boxes = []
        for xml_name in xml_names:
            xml_pth = os.path.join(anno_dir, xml_name)
            tree = et.parse(xml_pth)

            width = float(tree.findtext("./size/width"))
            height = float(tree.findtext("./size/height"))

            for obj in tree.findall("./object"):
                cls_name = obj.findtext("name")
                if cls_name in class_names:
                    xmin = float(obj.findtext("bndbox/xmin")) / width
                    ymin = float(obj.findtext("bndbox/ymin")) / height
                    xmax = float(obj.findtext("bndbox/xmax")) / width
                    ymax = float(obj.findtext("bndbox/ymax")) / height

                    box = [xmax - xmin, ymax - ymin]
                    boxes.append(box)
                else:
                    continue
        return np.array(boxes)

    anchors_txt = open(ANCHORS_TXT_PATH, "w")

    train_boxes = load_data(ANNOTATION_PATH, CLASS_NAMES)
    count = 1
    best_accuracy = 0
    best_anchors = []
    best_ratios = []

    for i in range(10):      ##### 可以修改，不要太大，否则时间很长
        anchors_tmp = []
        clusters = kmeans(train_boxes, k=CLUSTERS)
        idx = clusters[:, 0].argsort()
        clusters = clusters[idx]
        # print(clusters)

        for j in range(CLUSTERS):
            anchor = [round(clusters[j][0] * 640, 2), round(clusters[j][1] * 640, 2)]
            anchors_tmp.append(anchor)
            #print(f"Anchors:{anchor}")

        temp_accuracy = avg_iou(train_boxes, clusters) * 100
        #print("Train_Accuracy:{:.2f}%".format(temp_accuracy))

        ratios = (np.around(clusters[:, 0] / clusters[:, 1], decimals=2).tolist())
        ratios.sort()
        #print("Ratios:{}".format(ratios))
        #print(20 * "*" + " {} ".format(count) + 20 * "*")

        count += 1

        if temp_accuracy > best_accuracy:
            best_accuracy = temp_accuracy
            best_anchors = anchors_tmp
            best_ratios = ratios
    best_anchors_arr = np.array(best_anchors, dtype='uint16')

    best_anchors_arr = best_anchors_arr.reshape(best_anchors_arr.shape[0]*best_anchors_arr.shape[1])


    anchors_txt.write("Best Accuracy = " + str(round(best_accuracy, 2)) + '%' + "\r\n")
    anchors_txt.write("Best Anchors = " + str(best_anchors) + "\r\n")
    anchors_txt.write("Best Ratios = " + str(best_ratios))
    anchors_txt.close()
    with open("../models/yolov5s.yaml", "w") as f:
        f.write(STR_s.format(classnum, best_anchors_arr[0], best_anchors_arr[1], best_anchors_arr[2],
            best_anchors_arr[3], best_anchors_arr[4], best_anchors_arr[5], best_anchors_arr[6],
            best_anchors_arr[7], best_anchors_arr[8], best_anchors_arr[9], best_anchors_arr[10],
            best_anchors_arr[11], best_anchors_arr[12], best_anchors_arr[13], best_anchors_arr[14],
            best_anchors_arr[15], best_anchors_arr[16], best_anchors_arr[17]))
        #print("写入完毕！")
    with open(os.path.join(FILE_ROOT, "data.yaml"), "w", encoding='utf-8') as f1:
        f1.write(STR_yaml.format(str(os.path.join(vocpath,"train.txt")).replace("\\","/"),
                 str(os.path.join(vocpath, "val.txt")).replace("\\","/"), classnum, classesname))
    return best_anchors_arr, best_anchors, best_accuracy, best_ratios
