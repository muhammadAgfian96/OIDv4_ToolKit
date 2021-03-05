import os
from os.path import join as jpath, isdir
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import fileinput
import shutil


class OIDv6Handler:

    def __init__(self, oid_dataset_path = 'OID/Dataset', dst_folder='yolo_format', **kwargs):
        # get data
        self.args_get = {
            'command' : 'downloader',
            'classes' : ['Apple', 'Human_head'],
            'multiclasses': 0, # 1 if you wan all class in one folder, 0 for seperate in each folder base in class 
            'sub' : 'h',
            'type_csv': 'train',

            'image_IsOccluded': '',
            'image_IsTruncated': '', 
            'image_IsGroupOf' : 0, 
            'image_IsDepiction': 0, 
            'image_IsInside': '',

            'n_threads': 6,
            'limit': 1000,

            '--auto_split': True,
            '--split_ratio': (0.8, 0.1, 0.1)

        }

        self.args_get.update(**kwargs)

        # converting stuff
        self.dst_folder = dst_folder
        self.parent_oid_path = jpath(oid_dataset_path)
        self.type_dataset_path = [jpath(self.parent_oid_path, type_dataset) for type_dataset in os.listdir(self.parent_oid_path)]
        
        self.dst_folder =  jpath('OID', dst_folder)
        self.dst_folder_train =  jpath(self.dst_folder, 'train')
        self.dst_folder_valid =  jpath(self.dst_folder, 'valid')
        self.dst_folder_test =  jpath(self.dst_folder, 'test')

        self.child_dst_folder = [self.dst_folder_test, self.dst_folder_valid, self.dst_folder_train]
        for folder in self.child_dst_folder:
            if not os.path.exists(folder):
                os.makedirs(folder)
        # if multiclass
        
        # if not multiclass
        for type_data in self.type_dataset_path:
            # type_data = train, test, valid
            # os.makedirs(jpath(type_data, self.dst_folder))
            type_name_data = os.path.basename(type_data)
            raw_list_cls = os.listdir(type_data)
            classes = [cls for cls in raw_list_cls if '.' not in cls]
            self.cls2id ={}

            with open(jpath(self.dst_folder,type_name_data ,'_label.names'), 'w') as f:
                for idx, cls_name in enumerate(classes):
                    f.writelines(str(cls_name))
                    self.cls2id[cls_name] = idx
            print(self.cls2id, classes)



    # get data
    def cmd_get_data(self):
        my_args = ''
        for key in self.args_get.keys():
            if '--'  in key:
                continue

            value = self.args_get[key]
            if value == '':
                continue

            if key == 'command':
                my_args += f' {value}'
                continue

            if key == 'classes':
                my_args+= f' --{key}'
                for classes in value:
                    my_args += f' {classes}'
                continue

            my_args += f' --{key} {value}'

        if self.args_get['--auto_split']:
            for idx, ratio in enumerate(self.args_get['--split_ratio']):

                # train, valid, test
                if idx == 0:
                    new_args = my_args.replace("train", "train")
                    ls_args = new_args.split()
                    limit_new = int(ratio * self.args_get['limit'])
                    ls_args[-1] = str(limit_new)
                    new_args = " ".join(ls_args)
                    print(f'!python3 main.py {new_args}')

                elif idx == 1:
                    new_args = my_args.replace("train", "validation")
                    ls_args = new_args.split()
                    limit_new = int(ratio * self.args_get['limit'])
                    ls_args[-1] = str(limit_new)
                    new_args = " ".join(ls_args)
                    print(f'!python3 main.py {new_args}')

                elif idx == 2:
                    new_args = my_args.replace("train", "test")
                    ls_args = new_args.split()
                    limit_new = int(ratio * self.args_get['limit'])
                    ls_args[-1] = str(limit_new)
                    new_args = " ".join(ls_args)
                    print(f'!python3 main.py {new_args}')

    # converting
    def __read_label_oid(self, path_label):
        with open(path_label, 'r') as f:
            result = f.readlines()
            label = []
            for line in result:
                line = line.rstrip('\n')
                feat = line.split()
                if len(feat) != 5: print('[Error]', path); label.append([]);
                label.append(feat)
        return label

    def __convert2yolo(self, path_filename, features):
        parent_dir = os.path.dirname(path_filename)
        self.parent_dir = os.path.dirname(parent_dir)

        self.base_name = os.path.basename(path_filename).split('.')[0]
        self.image_path_src = jpath(self.parent_dir, self.base_name+'.jpg')
        image = cv2.imread(self.image_path_src)
        # print(image_path, os.path.isfile(image_path))
        h,w,c = image.shape
        #  Cx, Cy, w, h

        self.lines = []
        for feat in features:
            label_id = self.cls2id[feat[0]]
            xmin,ymin, xmax, ymax = float(feat[1]), float(feat[2]), float(feat[3]), float(feat[4])
            
            w_obj = xmax - xmin 
            y_obj = ymax - ymin 
            Cx = (xmax+xmin) / 2
            Cy = (ymax+ymin) / 2

            Cx = round(Cx/w, 6)        
            Cy = round(Cy/h, 6)
            w_obj = round(w_obj/w, 6)
            y_obj = round(y_obj/h, 6)
            self.lines.append(f'{label_id} {Cx} {Cy} {w_obj} {y_obj}')
        
        # write

    def __write_to_txt(self):
        with open(jpath(self.parent_dir, self.base_name+'.txt'), 'w') as f:
            for line in self.lines:
                f.writelines(line+'\n')
        self.label_path_src = jpath(self.parent_dir, self.base_name+'.txt')

    def __copy_to_dst_folder(self, type_data):
        shutil.copy2(self.image_path_src, jpath(self.dst_folder, type_data))
        shutil.copy2(self.label_path_src, jpath(self.dst_folder, type_data))
        self.image_path_src = None
        self.label_path_src = None

    def convert(self, to='yolo'):
        for full_type_data in self.type_dataset_path:
            the_type_data = os.path.basename(full_type_data)
            for cls in os.listdir(full_type_data):
                class_path = jpath(full_type_data, cls)
                label_path = jpath(class_path, 'Label')
                print(label_path)
                if isdir(label_path):
                    print('start converting')
                    for filename in tqdm(os.listdir(label_path)):
                        path_filename = jpath(label_path, filename)
                        # print('filename', path_filename)
                        features = self.__read_label_oid(path_filename)
                        # print('feature', features)
                        if to == 'yolo':
                            self.__convert2yolo(path_filename, features)
                            self.__write_to_txt()
                            self.__copy_to_dst_folder(the_type_data)
                        elif to == 'pascalVOC':
                            print('Not yet Implemented')
                            pass
                        elif to == 'COCO':
                            print('Not yet Implemented')
                            pass
                else:
                    print('not found anything ', full_type_data)




if __name__ == '__main__':
    args_get = {
        'command' : 'downloader',
        'classes' : ['Apple', 'Human_head'],
        'multiclasses': 0, # 1 if you wan all class in one folder, 0 for seperate in each folder base in class 
        'sub' : 'h',
        'type_csv': 'train',

        'image_IsOccluded': '',
        'image_IsTruncated': '', 
        'image_IsGroupOf' : 0, 
        'image_IsDepiction': 0, 
        'image_IsInside': '',

        'n_threads': 6,
        'limit': 1000,

        '--auto_split': True,
        '--split_ratio': (0.8, 0.1, 0.1)
    }

    dataset = OIDv6Handler(**args_get)
    dataset.cmd_get_data()
    # dataset.convert(to='yolo')
    # dataset.split('')