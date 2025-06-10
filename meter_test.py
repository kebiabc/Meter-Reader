import cv2
import os
from pathlib import Path
import math
import numpy as np  
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset,DataLoader, random_split
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import segmentation_models_pytorch as smp

METER_SHAPE = [512, 512] 
CIRCLE_CENTER = [256, 256] 
CIRCLE_RADIUS = 250
PI = math.pi
RECTANGLE_HEIGHT = 100
RECTANGLE_WIDTH = 785
TYPE_THRESHOLD = 40

METER_CONFIG = [{
    'scale_interval_value': 25.0 / 50.0,
    'range': .0,
    'unit': "(MPa)"
}, {
    'scale_interval_value': 1.6 / 32.0,
    'range': 1.6,
    'unit': "(MPa)"
}]

SEG_LABEL = {'background': 0, 'pointer': 1, 'scale': 2}


def pad_to_16_9(image):
    """
    Pad the image to 16:9 aspect ratio without changing the original aspect ratio.

    Param:
        image (np.ndarray): Input image in BGR format.

    Return:
        padded_image (np.ndarray): Padded image in BGR format.
    """
    h, w, _ = image.shape
    aspect_ratio = w / h

    # Calculate the new dimensions to achieve a 16:9 aspect ratio
    if aspect_ratio > 16 / 9:
        # Image is wider than 16:9, add padding to the top and bottom
        new_w = w
        new_h = int(w / (16 / 9))
        pad_h = (new_h - h) // 2
        padded_image = cv2.copyMakeBorder(image, pad_h, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        # Image is taller than 16:9, add padding to the left and right
        new_h = h
        new_w = int(h * (16 / 9))
        pad_w = (new_w - w) // 2
        padded_image = cv2.copyMakeBorder(image, 0, 0, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return padded_image

def roi_crop(image, results, scale_x, scale_y):
    """
    Crop the area of detected meter of original image

    Param:
        img (np.array):original image。
        det_results (list[dict]): detection results
        scale_x (float): the scale value in x axis
        scale_y (float): the scale value in y axis

    Returns:
        roi_imgs (list[np.array]): the list of meter images
        loc (list[int]): the list of meter locations
    
    """
    roi_imgs = []
    loc = []
    for result in results:
        bbox = result
        # ymin, xmin, ymax, xmax = [int(bbox[0] * scale_x), int(bbox[1] * scale_y), int(bbox[2] * scale_x), int(bbox[3] * scale_y)]
        xmin, ymin, xmax, ymax = [int(bbox[0] * scale_x), int(bbox[1] * scale_y), int(bbox[2] * scale_x), int(bbox[3] * scale_y)]
        sub_img = image[ymin:(ymax + 1), xmin:(xmax + 1), :]
        roi_imgs.append(sub_img)
        loc.append([xmin, ymin, xmax, ymax])
    return roi_imgs, loc

def roi_process(input_images, target_size, interp=cv2.INTER_LINEAR):
    """
    Prepare the roi image of detection results data
    Preprocessing the input data for segmentation task

    Param:
        input_images (list[np.array]):the list of meter images
        target_size (list|tuple): height and width of resized image, e.g [heigh,width]
        interp (int):the interp method for image reszing

    Returns:
        img_list (list[np.array]):the list of processed images
        resize_img (list[np.array]): for visualization
    
    """
    img_list = list()
    resize_list = list()
    for img in input_images:
        img_shape = img.shape
        scale_x = float(target_size[1]) / float(img_shape[1])
        scale_y = float(target_size[0]) / float(img_shape[0])
        resize_img = cv2.resize(img, None, None, fx=scale_x, fy=scale_y, interpolation=interp)
        resize_list.append(resize_img)
        resize_img = resize_img.transpose(2, 0, 1) / 255
        img_mean = np.array([0.5, 0.5, 0.5]).reshape((3, 1, 1))
        img_std = np.array([0.5, 0.5, 0.5]).reshape((3, 1, 1))
        resize_img -= img_mean
        resize_img /= img_std
        img_list.append(resize_img)
    return img_list, resize_list


def erode(seg_results, erode_kernel):
    """
    Erode the segmentation result to get the more clear instance of pointer and scale

    Param:
        seg_results (list[dict]):segmentation results
        erode_kernel (int): size of erode_kernel

    Return:
        eroded_results (list[dict]): the lab map of eroded_results
        
    """
    kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
    eroded_results = seg_results
    for i in range(len(seg_results)):
        eroded_results[i] = cv2.erode(seg_results[i].astype(np.uint8), kernel)
    return eroded_results


def circle_to_rectangle(seg_results):
    """
    Switch the shape of label_map from circle to rectangle

    Param:
        seg_results (list[dict]):segmentation results

    Return:
        rectangle_meters (list[np.array]):the rectangle of label map

    """
    rectangle_meters = list()
    for i, seg_result in enumerate(seg_results):
        label_map = seg_result

        # The size of rectangle_meter is determined by RECTANGLE_HEIGHT and RECTANGLE_WIDTH
        rectangle_meter = np.zeros((RECTANGLE_HEIGHT, RECTANGLE_WIDTH), dtype=np.uint8)
        for row in range(RECTANGLE_HEIGHT):
            for col in range(RECTANGLE_WIDTH):
                theta = PI * 2 * (col + 1) / RECTANGLE_WIDTH
                
                # The radius of meter circle will be mapped to the height of rectangle image
                rho = CIRCLE_RADIUS - row - 1
                y = int(CIRCLE_CENTER[0] + rho * math.cos(theta) + 0.5)
                x = int(CIRCLE_CENTER[1] - rho * math.sin(theta) + 0.5)
                rectangle_meter[row, col] = label_map[y, x]
        rectangle_meters.append(rectangle_meter)
    return rectangle_meters


def rectangle_to_line(rectangle_meters):
    """
    Switch the dimension of rectangle label map from 2D to 1D

    Param:
        rectangle_meters (list[np.array]):2D rectangle OF label_map。

    Return:
        line_scales (list[np.array]): the list of scales value
        line_pointers (list[np.array]):the list of pointers value

    """
    line_scales = list()
    line_pointers = list()
    for rectangle_meter in rectangle_meters:
        height, width = rectangle_meter.shape[0:2]
        line_scale = np.zeros((width), dtype=np.uint8)
        line_pointer = np.zeros((width), dtype=np.uint8)
        for col in range(width):
            for row in range(height):
                if rectangle_meter[row, col] == SEG_LABEL['pointer']:
                    line_pointer[col] += 1
                elif rectangle_meter[row, col] == SEG_LABEL['scale']:
                    line_scale[col] += 1
        line_scales.append(line_scale)
        line_pointers.append(line_pointer)
    return line_scales, line_pointers


def mean_binarization(data_list):
    """
    Binarize the data

    Param:
        data_list (list[np.array]):input data

    Return:
        binaried_data_list (list[np.array]):output data。

    """
    batch_size = len(data_list)
    binaried_data_list = data_list
    for i in range(batch_size):
        mean_data = np.mean(data_list[i])
        width = data_list[i].shape[0]
        for col in range(width):
            if data_list[i][col] < mean_data:
                binaried_data_list[i][col] = 0
            else:
                binaried_data_list[i][col] = 1
    return binaried_data_list


def locate_scale(line_scales):
    """
    Find location of center of each scale

    Param:
        line_scales (list[np.array]):the list of binaried scales value

    Return:
        scale_locations (list[list]):location of each scale

    """
    batch_size = len(line_scales)
    scale_locations = list()
    for i in range(batch_size):
        line_scale = line_scales[i]
        width = line_scale.shape[0]
        find_start = False
        one_scale_start = 0
        one_scale_end = 0
        locations = list()
        for j in range(width - 1):
            if line_scale[j] > 0 and line_scale[j + 1] > 0:
                if not find_start:
                    one_scale_start = j
                    find_start = True
            if find_start:
                if line_scale[j] == 0 and line_scale[j + 1] == 0:
                    one_scale_end = j - 1
                    one_scale_location = (one_scale_start + one_scale_end) / 2
                    locations.append(one_scale_location)
                    one_scale_start = 0
                    one_scale_end = 0
                    find_start = False
        scale_locations.append(locations)
    return scale_locations


def locate_pointer(line_pointers):
    """
    Find location of center of pointer

    Param:
        line_scales (list[np.array]):the list of binaried pointer value

    Return:
        scale_locations (list[list]):location of pointer

    """
    batch_size = len(line_pointers)
    pointer_locations = list()
    for i in range(batch_size):
        line_pointer = line_pointers[i]
        find_start = False
        pointer_start = 0
        pointer_end = 0
        location = 0
        width = line_pointer.shape[0]
        for j in range(width - 1):
            if line_pointer[j] > 0 and line_pointer[j + 1] > 0:
                if not find_start:
                    pointer_start = j
                    find_start = True
            if find_start:
                if line_pointer[j] == 0 and line_pointer[j + 1] == 0 :
                    pointer_end = j - 1
                    location = (pointer_start + pointer_end) / 2
                    find_start = False
                    break
        pointer_locations.append(location)
    return pointer_locations


def get_relative_location(scale_locations, pointer_locations):
    """
    Match location of pointer and scales

    Param:
        scale_locations (list[list]):location of each scale
        pointer_locations (list[list]):location of pointer

    Return:
        pointed_scales (list[dict]): a list of dict with:
                                     'num_scales': total number of scales
                                     'pointed_scale': predicted number of scales
            
    """
    pointed_scales = list()
    for scale_location, pointer_location in zip(scale_locations,
                                                pointer_locations):
        num_scales = len(scale_location)
        pointed_scale = -1
        if num_scales > 0:
            for i in range(num_scales - 1):
                if scale_location[i] <= pointer_location < scale_location[i + 1]:
                    pointed_scale = i + (pointer_location - scale_location[i]) / (scale_location[i + 1] - scale_location[i] + 1e-05) + 1
        result = {'num_scales': num_scales, 'pointed_scale': pointed_scale}
        pointed_scales.append(result)
    return pointed_scales


def calculate_reading(pointed_scales):
    """
    Calculate the value of meter according to the type of meter

    Param:
        pointed_scales (list[list]):predicted number of scales

    Return:
        readings (list[float]): the list of values read from meter
            
    """
    readings = list()
    batch_size = len(pointed_scales)
    for i in range(batch_size):
        pointed_scale = pointed_scales[i]
        # find the type of meter according the total number of scales
        if pointed_scale['num_scales'] > TYPE_THRESHOLD:
            reading = pointed_scale['pointed_scale'] * METER_CONFIG[0]['scale_interval_value']
        else:
            reading = pointed_scale['pointed_scale'] * METER_CONFIG[1]['scale_interval_value']
        readings.append(reading)
    return readings
  

def plot_result(img,readings,bboxs):
    '''
    Parameters
        img (np.ndarray) : image
        readings (array) : reading of detected meters
        bboxs(np.ndarray): BBOXs of detected meters
    
    Returns
        img (np.ndarray) : image with BBOX and readings
    '''
    for i in range(len(readings)):
        reading = readings[i]
        if reading >= 0:
            xmin,ymin,xmax,ymax = bboxs[i]

            img = cv2.rectangle(img,
                                (xmin,ymin),
                                (xmax,ymax),
                                (255, 51, 204),
                                3)
            img = cv2.putText(img,
                            f'{reading:.3f}',
                            (xmin,ymin-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (255,0,255),
                            6)
    return img

def pipeline(frame):
    '''
    Complete pipeline

    Image --> Detection --> Cropping --> Semantic-Segmentation --> Reading --> Plotting
    '''

    # Loading trained model {Traning is showed in segmentor.ipynb and detector.ipynb}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    segmentor = smp.Unet(encoder_name="efficientnet-b1", classes=3, activation=None).to(device)
    segmentor.load_state_dict(torch.load('meter_seg.pth', map_location=device))
    segmentor.eval()

    # 1. 初始化模型（在CPU上）
    detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT"
    )
    # 2. 修改分类头（仍在CPU上）
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, 2
    )
    # 3. 加载权重（指定映射到当前设备）
    detector.load_state_dict(torch.load('meter_detector.pth', map_location=device))
    # 4. 最后移动整个模型到设备
    detector = detector.to(device)
    detector.eval()


    # Getting Detection BBOXs
    frame = cv2.resize(frame,(1920,1080),cv2.INTER_AREA)
    transform = T.Compose([
    T.ToTensor(),  # 转为Tensor
    ])

    # 转为tensor，并添加 batch 维度
    input_tensor = transform(frame).to(device)  # [3, H, W]
    input_tensor = input_tensor.unsqueeze(0)    # [1, 3, H, W]

    # 进行预测
    with torch.no_grad():
        prediction = detector(input_tensor)[0]

    # 处理预测结果
    boxes = prediction['boxes'].cpu().numpy().astype(np.int64)
    scores = prediction['scores'].cpu().numpy()

    # 根据得分筛选
    selected_boxes = []
    for i in range(len(scores)):
        if scores[i] > 0.5:
            selected_boxes.append(boxes[i])
        else:
            break

    results = np.array(selected_boxes)
    print(results)
    # print(f"Detected {len(results)} meters.")
    # Cropping Meters
    roi_imgs,loc = roi_crop(frame,results,1,1)
    
    # Preprocess uneven cropped imgs to 256,256
    crop_img = []
    for roi_img in roi_imgs:
        resized = cv2.resize(roi_img,(256,256),cv2.INTER_AREA)
        crop_img.append(resized)

    plt.imshow(crop_img[0]); plt.title("Cropped"); plt.show()


    if len(crop_img) > 0:
        # Preprocess images for segmentation
        input_batch = []
        for img in crop_img:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Convert to tensor and permute dimensions
            tensor_img = torch.tensor(img_rgb, dtype=torch.float32).to(device)
            tensor_img = tensor_img.permute(2, 0, 1)  # Change from [H,W,C] to [C,H,W]
            # Normalize (same as roi_process)
            tensor_img = tensor_img / 255.0
            tensor_img = (tensor_img - 0.5) / 0.5
            input_batch.append(tensor_img)
        
        input_batch = torch.stack(input_batch)  # [B, C, H, W]

        # Getting Segmentation Maps
        with torch.no_grad():
            pred_logits = segmentor(input_batch)  # [B, 3, 256, 256]
            pred = torch.argmax(pred_logits, dim=1).cpu().numpy()  # [B, 256, 256]

        # Resize to 512x512 using nearest neighbor to preserve labels
        processed_seg = []
        for i in range(pred.shape[0]):
            resized = cv2.resize(
                pred[i].astype(np.uint8), 
                (512, 512), 
                interpolation=cv2.INTER_NEAREST
            )
            processed_seg.append(resized)
        pred = np.array(processed_seg)
        
        plt.imshow(pred[0]); plt.title("Segmentation"); plt.show()

        # Getting Reading from predicted Maps
        pred = erode(pred,2)
        rectangle_meters = circle_to_rectangle(pred)
        line_scales, line_pointers = rectangle_to_line(rectangle_meters)
        binaried_scales = mean_binarization(line_scales)
        binaried_pointers = mean_binarization(line_pointers)
        scale_locations = locate_scale(binaried_scales)
        pointer_locations = locate_pointer(binaried_pointers)
        pointed_scales = get_relative_location(scale_locations, pointer_locations)
        meter_readings = calculate_reading(pointed_scales)

        # Plotting reading and BBOXs on image
        plotted_img = plot_result(frame,meter_readings,results)
    
    else:
        plotted_img = frame
    return plotted_img
     

# img = cv2.imread('meter_test/20190822_93.jpg')   
img = cv2.imread('jingtian/5.jpg')   
padded_img = pad_to_16_9(img)
l = pipeline(padded_img)
plt.imshow(l)
plt.axis(False)
plt.show()