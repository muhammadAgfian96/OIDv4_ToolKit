<h1 align="center"> ~ OIDv6 ToolKit for Google Colabs ~ <br> (for yolov4 custom)</h1>

Hello, this is a tool for download OIDv6 and directly convert to yolo, pascalVOC and COCO.



# 1.0 Getting Started
cooming soon will update!

## 2.1 Download different classes in separated folders


```
main_folder
│   main.py
│
└───OID
    │   file011.txt
    │   file012.txt
    │
    └───csv_folder
    |    │   class-descriptions-boxable.csv
    |    │   validation-annotations-bbox.csv
    |
    └───Dataset
        |
        └─── test
        |
        └─── train
        |
        └─── validation
             |
             └───Apple
             |     |
             |     |0fdea8a716155a8e.jpg
             |     |2fe4f21e409f0a56.jpg
             |     |...
             |     └───Labels
             |            |
             |            |0fdea8a716155a8e.txt
             |            |2fe4f21e409f0a56.txt
             |            |...
             |
             └───Orange
                   |
                   |0b6f22bf3b586889.jpg
                   |0baea327f06f8afb.jpg
                   |...
                   └───Labels
                          |
                          |0b6f22bf3b586889.txt
                          |0baea327f06f8afb.txt
                          |...
```

## 2.2 Download multiple classes in a common folder
This option allows to download more classes, but in a common folder. Also the related notations are mixed together with
 the already explained format (the first element is always the name of the single class). In this way, with a simple
 dictionary it's easy to parse the generated label to get the desired format.

Again if we want to download Apple and Oranges, but in a common folder
  ```bash
   python3 main.py downloader --classes Apple Orange --type_csv validation --multiclasses 1
   ```

### Annotations

<img align="right" src="images/rectangle.png">

In the __original__ dataset the coordinates of the bounding boxes are made in the following way:

**XMin**, **XMax**, **YMin**, **YMax**: coordinates of the box, in normalized image coordinates. XMin is in [0,1], where 0 is the leftmost pixel, and 1 is the rightmost pixel in the image. Y coordinates go from the top pixel (0) to the bottom pixel (1).

However, in order to accomodate a more intuitive representation and give the maximum flexibility, every `.txt` annotation is made like:

`name_of_the_class    left    top     right     bottom`

where each coordinate is denormalized. So, the four different values correspond to the actual number of pixels of the related image.

If you don't need the labels creation use `--noLabels`.

### Optional Arguments
The annotations of the dataset has been marked with a bunch of boolean values. This attributes are reported below:
- **IsOccluded**: Indicates that the object is occluded by another object in the image.
- **IsTruncated**: Indicates that the object extends beyond the boundary of the image.
- **IsGroupOf**: Indicates that the box spans a group of objects (e.g., a bed of flowers or a crowd of people). We asked annotators to use this tag for cases with more than 5 instances which are heavily occluding each other and are physically touching.
- **IsDepiction**: Indicates that the object is a depiction (e.g., a cartoon or drawing of the object, not a real physical instance).
- **IsInside**: Indicates a picture taken from the inside of the object (e.g., a car interior or inside of a building).
- **n_threads**: Select how many threads you want to use. The ToolKit will take care for you to download multiple images in parallel, considerably speeding up the downloading process.
- **limit**: Limit the number of images being downloaded. Useful if you want to restrict the size of your dataset.
- **y**: Answer yes when have to download missing csv files.

Naturally, the ToolKit provides the same options as paramenters in order to filter the downloaded images.
For example, with:
  ```bash
   python3 main.py downloader -y --classes Apple Orange --type_csv validation --image_IsGroupOf 0
   ```
only images without group annotations are downloaded.

# 3.0 Download images from Image-Level Labels Dataset for Image Classifiction
The Toolkit is now able to acess also to the huge dataset without bounding boxes. This dataset is formed by 19,995 classes and it's already divided into train, validation and test. The command used for the download from this dataset is ```downloader_ill``` (Downloader of Image-Level Labels) and requires the argument ```--sub```. This argument selects the sub-dataset between human-verified labels ```h``` (5,655,108 images) and machine-generated labels ```m``` (8,853,429 images). An example of command is:
```bash
python3 main.py downloader_ill --sub m --classes Orange --type_csv train --limit 30
```
The previously explained commands ```Dataset```, ```multiclasses```, ```n_threads``` and ```limit``` are available.
The Toolkit automatically will put the dataset and the csv folder in specific folders that are renamed with a `_nl` at the end.
# Commands sum-up

|                    | downloader | visualizer | downloader_ill |                                                  |
|-------------------:|:----------:|:----------:|:--------------:|--------------------------------------------------|
|            Dataset |      O     |      O     |        O       | Dataset folder name                              |
|            classes |      R     |            |        R       | Considered classes                               |
|           type_csv |      R     |            |        R       | Train, test or validation dataset                |
|                  y |      O     |            |        O       | Answer yes when downloading missing csv files    |
|       multiclasses |      O     |            |        O       | Download classes toghether                       |
|           noLabels |      O     |            |                | Don't create labels                              |
|   Image_IsOccluded |      O     |            |                | Consider or not this filter                      |
|  Image_IsTruncated |      O     |            |                | Consider or not this filter                      |
|    Image_IsGroupOf |      O     |            |                | Consider or not this filter                      |
|  Image_IsDepiction |      O     |            |                | Consider or not this filter                      |
|     Image_IsInside |      O     |            |                | Consider or not this filter                      |
|          n_threads |      O     |            |        O       | Indicates the maximum threads number             |
|              limit |      O     |            |        O       | Max number of images to download                 |
|                sub |            |            |        R       | Human-verified or Machine-generated images (h/m) |

R = required, O = optional

# 4.0 Use the ToolKit to visualize the labeled images
The ToolKit is useful also for visualize the downloaded images with the respective labels.
```bash
   python3 main.py visualizer
   ```
  In this way the default `Dataset` folder will be pointed to search the images and labels automatically. To point
  another folder it's possible to use `--Dataset` optional argument.
```bash
   python3 main.py visualizer --Dataset desired_folder
   ```
Then the system will ask you which folder to visualize (train, validation or test) and the desired class.
Hence with `d` (next), `a` (previous) and `q` (exit) you will be able to explore all the images. Follow the menu for all the other options.

<p align="center">
  <img width="540" height="303" src="images/visualizer_example.gif">
</p>

# 5.0 Community Contributions
- [Denis Zuenko](https://github.com/zuenko) has added multithreading to the ToolKit and is currently working on the generalization and speeding up process of the labels creation
- [Skylion007](https://github.com/Skylion007) has improved labels creation reducing the runtime from O(nm) to O(n). That massively speeds up label generation
- [Alex March](https://github.com/hosaka) has added the limit option to the ToolKit in order to download only a maximum number of images of a certain class
- [Michael Baroody](https://github.com/mbaroody) has fixed the toolkit's visualizer for multiword classes

# Citation
Use this bibtex if you want to cite this repository:
```
@misc{OIDv4_ToolKit,
  title={Toolkit to download and visualize single or multiple classes from the huge Open Images v4 dataset},
  author={Vittorio, Angelo},
  year={2018},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/EscVM/OIDv4_ToolKit}},
}
```

# Reference
"[We don't need no bounding-boxes: Training object class detectors using only human verification](https://arxiv.org/abs/1602.08405)"Papadopolous et al., CVPR 2016.
