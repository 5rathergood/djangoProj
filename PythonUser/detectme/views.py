from django.shortcuts import render, redirect
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading

import argparse
import os
import sys
from pathlib import Path

# import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

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


# Create your views here.

class VideoCamera(object):
    def __init__(self):
        weights = "static/ds/yolov5/yolov5s.pt"
        source = "static/OTtest.mp4"
        data = "static/ds/yolov5/data/coco128.yaml"
        self.imgsz = [640, 640]
        self.device = ''
        line_thickness = 3
        self.half = False
        dnn = False

        source = str(source)
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.delay = int(1000 / fps)
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
        self.deepsort = DeepSort(deep_sort_model,
                                 max_dist=cfg.DEEPSORT.MAX_DIST,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)

        # Load model
        self.device = select_device(self.device)
        print(weights)
        print(self.device)
        print(dnn)
        print(data)
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.model = DetectMultiBackend(weights, self.device, dnn)
        # print('check')
        self.stride, names, pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        # print('check')
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        # Half
        self.half &= (
                             pt or jit or onnx or engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            self.model.model.half() if self.half else self.model.model.float()

        # print('check')
        self.video = cv2.VideoCapture(source)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        conf_thres = 0.5
        iou_thres = 0.45
        augment = False
        visualize = False
        classes = 0
        agnostic_nms = False
        max_det = 1000
        while True:
            (self.grabbed, self.frame) = self.video.read()

            #####################################################################################################
            # Padded resize
            img = letterbox(self.frame, self.imgsz, stride=self.stride)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = self.model(img, augment=augment, visualize=visualize)

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Process detections
            det = pred[0]

            # print(img.shape)

            if len(det):
                # tracker
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                outputs = self.deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), self.frame)
                # print()
                for j, (output, conf) in enumerate(zip(outputs, confs)):
                    bboxes = output[0:4]
                    id = output[4]
                    bboxes[2] = output[2] - output[0]
                    bboxes[3] = output[3] - output[1]

                    bboxes[0] = 2 * bboxes[0]
                    bboxes[1] = 2 * bboxes[1] - 20
                    bboxes[2] = 2 * bboxes[2]
                    bboxes[3] = 2 * bboxes[3]

                    point = (int(bboxes[0] + bboxes[2] / 2), int(bboxes[1] + bboxes[3] / 2))
                    cv2.rectangle(self.frame, bboxes, (0, 255, 0), 1)
                    cv2.line(self.frame, point, point, (255, 255, 255), 5)
                    cv2.putText(self.frame, str(id), (bboxes[0], bboxes[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                1)

                    # print(bboxes, id)

            #####################################################################################################

            #            bbox = (300, 100, 100, 150)
            #            point = (350, 175)
            #            cv2.rectangle(self.frame, bbox, (0, 255, 0), 1)
            #            cv2.line(self.frame, point, point, (255, 255, 255), 5)
            cv2.waitKey(self.delay)


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type : image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
def detectme(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        print("error!")
        pass


# using DB part
from .models import Traffic


def db_list(request):
    if request.method == 'POST':
        traffic_db = Traffic
        traffic_db.insertData(traffic_db, request.POST["p_id_text"])
        return redirect('db_list')
    else:
        traffic_list = Traffic.objects.all()
        return render(request, "home.html", {"traffic_list": traffic_list})


def home(request):
    return render(request, 'home.html')


def statistics(request):
    return render(request, 'statistics.html')


def analysis(request):
    return render(request, 'analysis.html')
