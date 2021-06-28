# Beetle Tracker

This tracker is for bury beetle experiments, and developed by [Prof. Sheng-Feng Shen's lab](https://ecology.lifescience.ntu.edu.tw/doku.php/en/sfshen/start). 

==gif==

It can identify 6 marks including `H, O, X, nn, ss, xx`.
|  0  |  1  |  2  |  3  |  4  |  5  |
|:---:|:---:|:---:|:---:|:---:|:---:|
|  H  |  O  |  X  | nn  | ss  | xx  |
|![](https://hackmd.io/_uploads/ryyk9VD7d.png)|![](https://hackmd.io/_uploads/BkcRjD7GO.png)|![](https://hackmd.io/_uploads/HJeoz_Qfd.png)|![](https://hackmd.io/_uploads/HJJryrPXd.png)|![](https://hackmd.io/_uploads/BkW8yd7MO.png)|![](https://hackmd.io/_uploads/H1yugYdmd.png)|


## Usage
1. Download 3 files from [here](https://drive.google.com/drive/folders/1mpe4q23KAurQ6MAhasBkWh5ahTfV2IOe?usp=sharing), including `BBCv1.pth`, `BBD1.2.pth`, and `simhei.ttf`, and put them into the `data` directory.
2. Put target `mp4` videos into `samples`, or you can set the directory path of videos through parameter `video_dir` in file `config.yml`.
3. Run the tracking script with command `python track.py` in terminal.

## Output
### JSON File
JSON files record tracking result for each frame. Data in each frame inculds:
- `boxes`: Bounding box coordinate in format `[left, top, right, bottom]`.
- `labels`: Marks on backs of the beetles.
- `label_IDs`: IDs of marks.
- `scores`: Tracking scores in range [0, 1].

### Video
Annotated videos with bounding boxes, labels, and tracking scores.

## Setting
Parameters can be edited in `config.yml`.
- `output_jsons`: To output JSON files.
- `output_videos`: To output videos.
- `center_crop_4x3`: Center crop videos with 4:3 aspect ratio while tracking.
- `video_dir`: Directory path of the target videos.
- `output_dir`: Output directory path.