from PIL import Image, ImageDraw, ImageFont
import numpy as np
import colorsys
import random



def set_colors(class_n, satu=1., light=1.):
    hsv_tuples = [(x/class_n, satu, light) for x in range(class_n)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x),hsv_tuples))
    colors = list(map(lambda x: (int(x[0]*255),int(x[1]*255),int(x[2]*255)),colors))
    random.shuffle(colors)
    return colors


def paint_mask(mask_path, colors, alpha=.8, background=None):
    colors = np.array(colors)
    img_array = np.array(Image.open(mask_path))
    colors = np.array(colors).astype(int)
    mask_array = colors[img_array.astype(int)]
    img_mask = Image.fromarray(np.uint8(mask_array))
    img_mask.putalpha(int(255*alpha))
    
    if background == None:
        return img_mask
    else:
        img = Image.open(background)
        img_mask = img_mask.resize(img.size, Image.NEAREST)
        img.paste(img_mask, mask=img_mask)
        return img
      
    
def mark_boxes(
        image, 
        boxes:list, 
        colors:list = None,
        tags:list = None,
        line_thick = 3,
        font_path = None,
        text_size = 30
    ):
    """Draw bounding boxes by given detection result"""
    
    img = image.copy()
    if font_path is not None:
        font = ImageFont.truetype(font=font_path, size=text_size)
    else:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)
 
    for i, box in enumerate(boxes):
        # set color
        if colors is None:
            color = (255, 0, 255)
        else:
            color = colors[i]
            
        # box
        draw.rectangle(box, fill=None, outline=color, width=line_thick)
        
        # tag
        tag = ''
        if tags is not None:
            tag += f'{tags[i]} '
        if tag != '':
            text_bar = [box[0], box[1]-text_size, box[0]+len(tag)*text_size//2, box[1]]
            draw.rectangle(text_bar, fill=color, outline=color, width=line_thick)
            draw.text(text_bar[:2], tag, fill=(0,0,0), font=font)
    return img



def mark_centers(image, detection, colors=None, show_text=True, point_size=3, text_size=1):
    """Draw bounding boxes by given detection result
    
    detection: list of detection result in format [[left, top, right, bottom, class_no, score, class_name], ...]
    """
    img = image.copy()
    if show_text:
        text_size = np.floor(text_size * 3e-2 * np.shape(img)[1] + 0.5).astype('int32')
        font = ImageFont.truetype(font='simhei.ttf', size=text_size)

    draw = ImageDraw.Draw(img)
    # sort bboxes by score
    for bbox in sorted(detection, key=lambda s: s[5]):
        # get center from bbox
        x = (bbox[0]+bbox[2]) // 2
        y = (bbox[1]+bbox[3]) // 2
        # set colors
        if colors is None:
            color = (255, 0, 255)
        else:
            color = colors[bbox[4]]
        point_loc = [x-point_size, y-point_size, x+point_size, y+point_size]
        draw.ellipse(point_loc, fill=color, outline=None, width=1)
        # tag
        if show_text:
            tag = '{}'.format(bbox[6])
            text_loc = [x+2*point_size, y]
            draw.text(text_loc, tag, fill=color, font=font)
    del draw
    return img