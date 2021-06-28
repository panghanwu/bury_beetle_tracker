from tqdm import tqdm
from glob import glob
from PIL import Image
import torchvision.transforms as tsf
import numpy as np
import cv2
import os
import yaml
import json
import csv

from utils import painter
import app


def crop_objects(pil_image, boxes:list):
    """
    Args:
        pil_image: Input PIL image.
        boxes: List of boxes formated in [left, top, right, bottom]
    """
    objects = list()
    for b in boxes:
        objects.append(pil_image.crop(b))
    return objects


# load config
with open('config.yml') as yf:
    config = yaml.full_load(yf)

video_list = sorted(glob(config['video_dir'] + '*'))
output_dir = config['output_dir']

# create output dit
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# load model
yolo_detector = app.Detector(config['Detector'])
yolo_detector.init()
resnet_marker = app.Marker(config['Classifier'])
resnet_marker.init()
tracker = app.Tracker(config['Tracker'])

# load label dict
label_dict = config['Classifier']['class_names']

# set colors
color_list = painter.set_colors(len(label_dict)-1)
color_list += [(255,255,255)]
color_dict = {cl: co for cl, co in zip(label_dict.values(), color_list)}


for video_path in video_list:
    video = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    total_f = int(video.get(7))
    tracker.reset()
    frame_dict = dict()
    
    # get w, h
    if config['center_crop_4x3']:
        w, h = int(video.get(4)*4/3), int(video.get(4))
    else:
        w, h = int(video.get(3)), int(video.get(4))

    if config['output_video']:
        output_name = output_dir + f'track_{video_name}'

        # set video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(output_name, fourcc, video.get(5), (w, h))
    
    with tqdm(desc=video_name, total=total_f) as pbar:

        while video.isOpened():
            f_no = int(video.get(1))
            success, frame = video.read()
            
            if success:
                # get frames to list
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

                if config['center_crop_4x3']:
                    frame = tsf.CenterCrop((h,w))(frame)
                
                # detect
                detection = yolo_detector.detect(frame)
                boxes = [d[:4] for d in detection]
                        
                # track
                crops = crop_objects(frame, boxes)
                class_codes = resnet_marker.mark_batch(crops, return_raw=True)
                tracks = tracker.track(boxes, class_codes)

                label_ids = [str(t[0]) for t in tracks]
                labels = [label_dict[int(i)] for i in label_ids]
                scores = [round(t[1], 2) for t in tracks]
                boxes = [list(map(int, t[2])) for t in tracks]

                if config['output_json']:
                    frame_dict[str(f_no)] = {
                        'boxes': boxes,
                        'labels': labels,
                        'label_IDs': label_ids,
                        'scores': scores
                    }

                if config['output_video']:
                    # set colors
                    colors = [color_list[t[0]] for t in tracks]
                    tags = [f'{t} {s:.2f}' for t, s in zip(labels, scores)]

                    # draw boxes
                    frame = painter.mark_boxes(
                        frame, 
                        boxes, 
                        colors = colors, 
                        tags = tags,
                        font_path = config['text_font_path']
                    )

                    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                    output.write(frame)
                    
                pbar.update(1)
            else:
                break

        if config['output_video']:
            video.release()
            output.release()

        if config['output_json']:
            json_name = output_dir + f'{os.path.splitext(video_name)[0]}.json'
            with open(json_name, 'w') as jf:
                json.dump(frame_dict, jf, indent=4)
        
        if config['output_csv']:
            # label_ids, labels, score, boxes
            csv_data = list()

            for data in frame_dict.values():

                # create template
                csv_line = [None]*6*7
                for i in range(6):
                    csv_line[7*i] = i
                    csv_line[7*i+1] = label_dict[i]

                for i in range(len(data['labels'])):
                    ndx = csv_line.index(data['labels'][i])
                    csv_line[ndx+1] = data['scores'][i]
                    csv_line[ndx+2:ndx+6] = data['boxes'][i]

                csv_data.append(csv_line)

            csv_name = output_dir + f'{os.path.splitext(video_name)[0]}.csv'
            with open(csv_name, 'w') as cf:
                write = csv.writer(cf)
                write.writerows(csv_data)

print('Completed!')