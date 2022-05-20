
import argparse
import os
import sys
from pathlib import Path

# import cv2
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from django.db import IntegrityError
from datetime import date, time, datetime
from detectme.models import TodayTraffic, TodayRecord, lineRecord

# import yolov5
ROOT = "static/ds"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

YOLOROOT = "static/ds/yolov5"
if str(YOLOROOT) not in sys.path:
    sys.path.append(str(YOLOROOT))  # add YOLOROOT to PATH

sys.path.append("..")

from PythonUser.static.ds.yolov5.models.common import DetectMultiBackend
from PythonUser.static.ds.yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, letterbox
from PythonUser.static.ds.yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow,
                                                       check_requirements, colorstr,
                                                       increment_path, non_max_suppression, print_args, scale_coords,
                                                       strip_optimizer,
                                                       xyxy2xywh)
from PythonUser.static.ds.yolov5.utils.plots import Annotator, colors, save_one_box
from PythonUser.static.ds.yolov5.utils.torch_utils import select_device, time_sync

# import deep_sort
DSDIR = "static/ds/deep_sort/deep/reid"
if str(DSDIR) not in sys.path:
    sys.path.append(str(DSDIR))  # add ROOT to PATH

from PythonUser.static.ds.deep_sort.utils.parser import get_config
from PythonUser.static.ds.deep_sort.deep_sort import DeepSort

import line_cap as line_cap
import threading
from django.utils import timezone

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



def ObjectTrack(q, line_check):
    weights = "static/ds/yolov5/yolov5s.pt"
    source = "static/OTtest.mp4"
    #source = "rtsp://10.20.3.121:8080/h264_ulaw.sdp"
    #source = 'rtsp://admin:5glfuwhgdk@192.168.0.11:554/ISAPI/streaming/channels/101？transportmode=multicast'
    
    data = "static/ds/yolov5/data/coco128.yaml"
    imgsz = [640, 640]
    device = ''
    line_thickness = 3
    half = False
    dnn = False

    source = str(source)
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)
    obj_det = True
    det_count = 0

    detnum = 0

    tracker = []

    config_deepsort = 'static/ds/deep_sort/configs/deep_sort.yaml'
    deep_sort_model = 'osnet_x0_25'

    # w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                             max_dist=cfg.DEEPSORT.MAX_DIST,
                             max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                             max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                             nn_budget=cfg.DEEPSORT.NN_BUDGET,
                             use_cuda=True)

    # Load model
    device = select_device(device)
    print(weights)
    print(device)
    print(dnn)
    print(data)
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = DetectMultiBackend(weights, device, dnn)
    # print('check')
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    # print('check')
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (
                         pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # print('check')
    video = cv2.VideoCapture(source)
    (grabbed, frame) = video.read()
    
    conf_thres = 0.5
    iou_thres = 0.45
    augment = False
    visualize = False
    classes = 0
    agnostic_nms = False
    max_det = 1000

    obj_num = []
    obj_tail = []
    
    lines = []
    line_cap.load_lines(lines)
    #cap.line_load(lines)
    #line_lists = line_list.objects.all()        

    #line_check = True
    
    
    while True:
        (grabbed, frame) = video.read()
        #print(grabbed)
        if not grabbed:
            video = cv2.VideoCapture(source)
            continue


        # line management, run once
        #print(line_check)
        check = False
        if not line_check.empty():
            check = line_check.get()
            while not line_check.empty():
                line_check.get()
        if check:
            #print(len(im0s[0]))
            line_cap.line_manage(frame, lines)
            print(lines)
            #print(type(im0s))

        # Padded resize
        img = letterbox(frame, imgsz, stride=stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=augment, visualize=visualize)

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process detections
        det = pred[0]

        # print(img.shape)
        line_cap.draw_lines(frame, lines)
        line_cap.line_numbering(frame, lines)

        if len(det):
            # tracker
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)
            # print()
            for j, (output, conf) in enumerate(zip(outputs, confs)):
                bboxes = output[0:4]
                id = output[4]
                
                #object list
                if id not in obj_num:
                    obj_num.append(id)
                    obj_tail.insert(obj_num.index(id), [])
                    
                bboxes[2] = output[2] - output[0]
                bboxes[3] = output[3] - output[1]
                
                bboxes[0] = 2 * bboxes[0]
                bboxes[1] = 2 * bboxes[1] - 20
                bboxes[2] = 2 * bboxes[2]
                bboxes[3] = 2 * bboxes[3]
                

                point = (int(bboxes[0] + bboxes[2] / 2), int(bboxes[1] + bboxes[3] / 2))
                cv2.rectangle(frame, bboxes, (0, 255, 0), 2)
                cv2.line(frame, point, point, (255, 255, 255), 5)
                cv2.putText(frame, str(id), (bboxes[0], bboxes[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            2)

                #object tail track
                obj_tail[obj_num.index(id)].append(point)
                if len(obj_tail[obj_num.index(id)]) > 30:
                    del obj_tail[obj_num.index(id)][0]

                for i in range(len(obj_tail[obj_num.index(id)]) - 1):
                    tail_point1 = obj_tail[obj_num.index(id)][i]
                    tail_point2 = obj_tail[obj_num.index(id)][i+1]
                    cv2.line(frame, tail_point1, tail_point2, (0, 0, 255), 2)

                #line counting
                line_cap.check_cross(id, obj_tail[obj_num.index(id)], lines)
                
        line_cap.line_count(frame, lines)
        q.put(frame)
        cv2.waitKey(delay)
        
        #datas for DB
        alley_people = []
        for i, on_mouse, line, out_count, in_count in lines:
           # i             #라인 번호
           # len(in_count) #해당 라인으로 입장한 유동 인구 수
            if i == 0:
                alley_people += in_count

            elif i == 1:
                alley_people += in_count
                alley_people = list(set(alley_people))
                
                lineRecord.objects.create(
                    line_id = int(i),
                    people_count = int(len(alley_people)),
                    cross_time = timezone.now()
                )
                for person_id in alley_people:
                    #TodayTraffic & TodayRecord DB insert
                    p_id = person_id
                    today = datetime.today()
                    today_date = today.date()
                    today_time = today.strftime('%H:%M:%S')
                    today_hour = int(today.strftime('%H'))
                    index = 'time_' + str(today_hour + 1)
                    try:
                        TodayTraffic.objects.create(person_id=p_id, date=today_date, time=today_time)
                    except IntegrityError:
                        pass
                    else:
                        print("this is hour ", today_hour)
                        target_row = TodayRecord.objects.first()
                        target_row.__dict__[index] += 1
                        target_row.save()
                #print(alley_people)

            else:
               lineRecord.objects.create(
                   line_id = int(i),
                   people_count = int(len(in_count)),
                   cross_time = timezone.now()
               )
