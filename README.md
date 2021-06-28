# Beetle Tracker

Tracker for bury beetle experiments, developed by [Prof. Sheng-Feng Shen's lab](https://ecology.lifescience.ntu.edu.tw/doku.php/en/sfshen/start). 

![](https://github.com/panghanwu/bury_beetle_tracker/blob/main/meterials/example.gif)

It can identify 6 marks including `H, O, X, nn, ss, xx`.
|  0  |  1  |  2  |  3  |  4  |  5  |
|:---:|:---:|:---:|:---:|:---:|:---:|
|  H  |  O  |  X  | nn  | ss  | xx  |
|![](https://github.com/panghanwu/bury_beetle_tracker/blob/main/meterials/m0.png)|![](https://github.com/panghanwu/bury_beetle_tracker/blob/main/meterials/m1.png)|![](https://github.com/panghanwu/bury_beetle_tracker/blob/main/meterials/m2.png)|![](https://github.com/panghanwu/bury_beetle_tracker/blob/main/meterials/m3.png)|![](https://github.com/panghanwu/bury_beetle_tracker/blob/main/meterials/m4.png)|![](https://github.com/panghanwu/bury_beetle_tracker/blob/main/meterials/m5.png)|


## Usage
1. Download 2 files from [here](https://drive.google.com/drive/folders/1mpe4q23KAurQ6MAhasBkWh5ahTfV2IOe?usp=sharing), including `BBCv1.pth` and `BBD1.2.pth`, and put them into the `data` directory.
2. Set the path of the target `mp4` videos through parameter `video_dir` in file `config.yml`.
3. Run the tracking script with command `python track.py` in terminal.

## Output
### JSON File
JSON files record tracking result for each frame. Data in each frame includes:
- `boxes`: Bounding box coordinate in format `[left, top, right, bottom]`.
- `labels`: Marks on backs of the beetles.
- `label_IDs`: IDs of marks.
- `scores`: Tracking scores in range [0, 1].

### CSV File
The Nth line in a CSV file corresponds to the Nth frame of the corresponding video. Each line is consist of the 6 marks in format:
`label_id, label, score, left, top, right, bottom`
The score and box will be empty if there is no beetle with the certain mark be detected.

### Video
Annotated videos with bounding boxes, labels, and tracking scores.

## Setting
Parameters can be edited in `config.yml`.
- `output_json`: To output JSON files.
- `output_csv`: To output CSV files.
- `output_video`: To output videos.
- `center_crop_4x3`: Center crop videos with 4:3 aspect ratio while tracking.
- `video_dir`: Directory path of the target videos.
- `output_dir`: Result directory path.
