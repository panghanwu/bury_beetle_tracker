from PIL import Image
from torchvision import ops
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
import torchvision.transforms as tsf
import torch.nn as nn
import numpy as np
import torch

# local module
from nets.yolo4 import YoloBody
from utils.utils import (
    DecodeBox,
    non_max_suppression
)
from nets import classification


class Detector:
    """
    Function:
        detect: Input an image and return list of inferred bounding boxes in format:
                [top, left, bottom, right, class_no, score]
    """
    def __init__(self, config):
        self.model_name = config['model_name']
        self.state_dict_path = config['state_dict_path']
        self.class_names = config['class_names']
        self.input_size = config['input_size']
        self.anchors = config['anchors']
        self.confidence = config['confidence']
        self.iou = config['iou']
        self.device = config['device']
        self.grayscale = config['grayscale']
        
    # set/reset model
    def init(self):   
        device = torch.device(self.device)
        
        # load model
        self.yolo = YoloBody(len(self.anchors)//3, len(self.class_names)).eval()
        state_dict = torch.load(self.state_dict_path)
        self.yolo.load_state_dict(state_dict)
        self.yolo = self.yolo.to(self.device)
        
        # bbox decoder
        self.yolo_decodes = []
        self.anchors_mask = [[6,7,8],[3,4,5],[0,1,2]]
        for i in range(3):
            self.yolo_decodes.append(DecodeBox(
                np.array(self.anchors)[self.anchors_mask[i]],
                len(self.class_names),  
                self.input_size
            )
        )
             
        print(f'Load {self.model_name} and set device to "{self.device}" successfully!')
        
    def detect(self, pil_image):
        w, h = pil_image.size

        if self.grayscale:
            pil_image = pil_image.convert('L')
        # set channels to 3
        pil_image = pil_image.convert('RGB')
        # resize to fit input
        pil_image = pil_image.resize(self.input_size, Image.BICUBIC)
        # to tensor
        image = tsf.ToTensor()(pil_image)
        # add batch dim
        image = image.unsqueeze(0)
        
        self.result = []  # create result container
        
        # turn off autograd
        with torch.no_grad():
            image = image.to(self.device)
                
            # detect via model
            outputs = self.yolo(image)
            
            # decode bbox
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))
                
            # NMS
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(
                output, 
                len(self.class_names),
                conf_thres = self.confidence,
                nms_thres = self.iou
            )
            
            # if no object is detected
            if batch_detections != [None]:
                batch_detections = batch_detections[0].cpu().numpy()
            
                # filter bbox under threshold
                top_index = batch_detections[:, 4]*batch_detections[:, 5] > self.confidence
                top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
                top_label = np.array(batch_detections[top_index,-1], np.int32)
                top_bboxes = np.array(batch_detections[top_index,:4])
                top_xmin = np.expand_dims(top_bboxes[:,0],-1)
                top_ymin = np.expand_dims(top_bboxes[:,1],-1)
                top_xmax = np.expand_dims(top_bboxes[:,2],-1)
                top_ymax = np.expand_dims(top_bboxes[:,3],-1)

                # align bbox back to origin size
                top_xmin = top_xmin/self.input_size[0] * w
                top_ymin = top_ymin/self.input_size[1] * h
                top_xmax = top_xmax/self.input_size[0] * w
                top_ymax = top_ymax/self.input_size[1] * h
                boxes = np.concatenate([top_xmin,top_ymin,top_xmax,top_ymax], axis=-1)

                # gather bbox to list (top, left, bottom, right, class_no, score)
                for i, class_no in enumerate(top_label):
                    score = top_conf[i]
                    self.result.append(boxes[i].tolist() + [class_no, score])
        return self.result
        
    def detect_batch(self, pil_image_list):
        batch_size = len(pil_image_list)
        images = list()
        for pil_image in pil_image_list:
            w, h = pil_image.size

            if self.grayscale:
                pil_image = pil_image.convert('L')
            # set channels to 3
            pil_image = pil_image.convert('RGB')
            # resize to fit input
            pil_image = pil_image.resize(self.input_size, Image.BICUBIC)
            # to tensor
            image = tsf.ToTensor()(pil_image)
            # add batch dim
            image = image.unsqueeze(0)
            images.append(image)
            
        images = torch.cat(images)
        self.result = list()  # create result container
        
        # turn off autograd
        with torch.no_grad():
            images = images.to(self.device)
                
            # detect via model
            outputs = self.yolo(images)
            
            for b in range(batch_size):
                # decode bbox
                output_list = list()
                for i in range(3):
                    output_list.append(self.yolo_decodes[i](outputs[i][b:b+1]))

                # NMS
                output = torch.cat(output_list, 1)
                batch_detections = non_max_suppression(
                    output, 
                    len(self.class_names),
                    conf_thres = self.confidence,
                    nms_thres = self.iou
                )
                
                temp_result = list()
                # if no object is detected
                if batch_detections != [None]:
                    batch_detections = batch_detections[0].cpu().numpy()

                    # filter bbox under threshold
                    top_index = batch_detections[:, 4]*batch_detections[:, 5] > self.confidence
                    top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
                    top_label = np.array(batch_detections[top_index,-1], np.int32)
                    top_bboxes = np.array(batch_detections[top_index,:4])
                    top_xmin = np.expand_dims(top_bboxes[:,0],-1)
                    top_ymin = np.expand_dims(top_bboxes[:,1],-1)
                    top_xmax = np.expand_dims(top_bboxes[:,2],-1)
                    top_ymax = np.expand_dims(top_bboxes[:,3],-1)

                    # align bbox back to origin size
                    top_xmin = top_xmin/self.input_size[0] * w
                    top_ymin = top_ymin/self.input_size[1] * h
                    top_xmax = top_xmax/self.input_size[0] * w
                    top_ymax = top_ymax/self.input_size[1] * h
                    boxes = np.concatenate([top_xmin,top_ymin,top_xmax,top_ymax], axis=-1)

                    # gather bbox to list (top, left, bottom, right, class_no, score)
                    for i, class_no in enumerate(top_label):
                        score = top_conf[i]
                        temp_result.append(boxes[i].tolist() + [int(class_no), float(score)])
                        
                self.result.append(temp_result)
        return self.result
    

class Marker:
    
    def __init__(self, config):
        self.model_name = config['model_name']
        self.state_dict_path = config['state_dict_path']
        self.device = config['device']
        self.input_size = config['input_size']
        self.class_names = config['class_names']
        self.threshold = config['threshold']
        
    def _cook_input(self, pil_image):
        """
        Input an PIL image then renturn a processed tensor image.
        """
        pil_image = pil_image.resize(self.input_size)
        pil_image = pil_image.convert('L').convert('RGB')
        ts_image = tsf.ToTensor()(pil_image).unsqueeze(0)
        return ts_image.to(self.device)
    
    def _cook_results(self, predictions):
        upper_thres = predictions>self.threshold
        success_mask = torch.sum(upper_thres, dim=1)==1
        unknown_mask = torch.sum(upper_thres, dim=1)!=1

        scores, class_no = torch.max(predictions, dim=1)
        class_no[unknown_mask] = 6
        scores[unknown_mask] = 0.
        
        results = [self.class_names[i] for i in class_no.cpu().tolist()]
        return results, scores.cpu().tolist()
    
    def init(self):
        device = torch.device(self.device)
        
        # load model
        self.marker = classification.marker_classifier(len(self.class_names))
        state_dict = torch.load(self.state_dict_path)
        self.marker.load_state_dict(state_dict)
        self.marker = self.marker.to(self.device).eval()
        print(f'Load {self.model_name} and set device to "{self.device}" successfully!')
        
    def mark(self, pil_image, return_raw=False):
        ts_image = self._cook_input(pil_image)
        with torch.no_grad():
            pred = self.marker(ts_image)
            
        if return_raw:
            return pred.cpu().tolist()
        else:
            [result], [score] = self._cook_results(pred)
            return result, score
    
    def mark_batch(self, pil_images:list, batch_size=6, return_raw=False):
        """
        Args:
            pil_images: List of PIL images.
            batch_size: Maximun size for a batch.
        Return:
            All codes for each image in tensor.
        """
        if pil_images == list():
            results = list()
            scores = list()
        else:
            ts_images = list()
            for img in pil_images:
                ts_images.append(self._cook_input(img))
            ts_images = torch.cat(ts_images)

            results = list()
            scores = list()
            head = 0
            rest = len(pil_images)
            while rest > 0:

                if rest < batch_size:
                    tail = len(pil_images)
                else:
                    tail = head + batch_size

                with torch.no_grad():
                    batch_preds = self.marker(ts_images[head:tail])
                    
                if return_raw:
                    results += batch_preds.cpu().tolist()
                else:
                    batch_results, batch_scores = self._cook_results(batch_preds)
                    results += batch_results
                    scores += batch_scores
                    
                head = tail
                rest = len(pil_images) - head
                    
        if return_raw:
            return results
        else:
            return results, scores


class Tracker:
    
    def __init__(self, config):
        
        self.num_classes = config['num_classes']
        
        # initialize
        self.footprints = [
            [i for i in range(config['num_classes'])],  # class IDs
            [None]*config['num_classes'],  # boxes
            [None]*config['num_classes'],  # feature codes
            [False]*config['num_classes']  # mask
        ]
        
        # parameters
        self.w = config['weights']
        self.thrs = config['score_threshold']
    
    def reset(self):
        self.footprints = [
            [i for i in range(self.num_classes)],  # class IDs
            [None]*self.num_classes,  # boxes
            [None]*self.num_classes,  # feature codes
            [False]*self.num_classes  # mask
        ]
    
    def track(self, boxes:list, classes:list):
        """
        Args:
            boxes: List of bounding boxes in sequence in format:
                   [left, top, right, bottom].
            classes: List of class codes in sequence.
        Return:
            List of tracks in format:
            [class ID, score, box]
        """
        tracks = list()
    
        if boxes!=list():

            # class criterion        
            score_matrix = torch.Tensor(classes)
            
            if any(self.footprints[3]):
                
                old_boxes = list()
                for box, ftr in zip(self.footprints[1], self.footprints[2]):
                    if box is not None:
                        old_boxes.append(box)
                
                old_boxes = torch.Tensor(old_boxes)
                new_boxes = torch.Tensor(boxes)
                
                # criteria
                iou_matrix = ops.box_iou(new_boxes, old_boxes)
                
                score_matrix[:, self.footprints[3]] = self.w[0]*iou_matrix + \
                                                      self.w[1]*score_matrix[:, self.footprints[3]]
            
            # match
            seq_id, cls_id = linear_sum_assignment(score_matrix, maximize=True)
            
            for s, c in zip(seq_id, cls_id):
                if score_matrix[s,c] > self.thrs:
                
                    # footprint
                    self.footprints[1][c] = boxes[s]
                    self.footprints[3][c] = True

                    # tracks
                    tracks.append([c, score_matrix[s,c].tolist(), boxes[s]])
        
        return tracks