from datetime import date, time, datetime
from django.shortcuts import render, redirect
from django.views.decorators import gzip
from django.http import StreamingHttpResponse, JsonResponse
from django.db import IntegrityError
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

                    #insert to DB
                    p_id = id
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
from .models import TodayTraffic, TodayRecord, Record

# def db_list(request):
#     if request.method == 'POST':
#         traffic_db = TodayTraffic
#         traffic_db.insertData(traffic_db, request.POST["p_id_text"])
#         return redirect('db_list')
#     else:
#         traffic_list = TodayTraffic.objects.all()
#         return render(request, "home.html", {"traffic_list": traffic_list})
#
#
# def home(request):
#     return render(request, 'home.html')
#
#
# def statistics(request):
#     traffic_list = TodayTraffic.objects.all()
#     return render(request, 'statistics.html', {"traffic_list": traffic_list})
#
#
# def analysis(request):
#     return render(request, 'analysis.html')


def home(request):
    # if request.method == 'POST':
    #     p_id = request.POST["p_id_text"]
    #     today = datetime.today()
    #     today_date = today.date()
    #     today_time = today.strftime('%H:%M:%S')
    #     try:
    #         TodayTraffic.objects.create(person_id=p_id, date=today_date, time=today_time)
    #     except IntegrityError:
    #         pass
    #     return redirect('home')
    # else:

    #최초 실행시 today_record_list가 비어있다면 default row를 하나 생성
    today_record_list = TodayRecord.objects.all()
    if len(today_record_list) == 0:
        first_low = TodayRecord.objects.create()
        first_low.save()

    #TodayTraffic.objects.bulk_update()
    traffic_list = TodayTraffic.objects.all()

    traffic_list = [traffic.get_person_id() for traffic in traffic_list]
    #record_list = Record.objects.all()
    #record_list = [record.get_values() for record in record_list]
    today_record_list = [today_record.get_values() for today_record in today_record_list]
    print(today_record_list)

    return render(request, "home.html", {"traffic_list": traffic_list, "today_record_list": today_record_list})


def db_list(request):
    today_record_list = TodayRecord.objects.all()
    if len(today_record_list) == 0:
        first_low = TodayRecord.objects.create()
        first_low.save()

    today_record_list = [today_record.get_values() for today_record in today_record_list]

    return JsonResponse(today_record_list[0])


def statistics(request):
    #일별 그래프
    record_day_list = Record.objects.all()
    record_day_list = [record.get_count_and_month() for record in record_day_list]

    #월별 그래프
    record_month_list = [{'all_count': 0, 'month': '0'} for _ in range(0, 12)]

    record_month_db_list = [
        Record.objects.all().filter(count_date__month='01'),
        Record.objects.all().filter(count_date__month='02'),
        Record.objects.all().filter(count_date__month='03'),
        Record.objects.all().filter(count_date__month='04'),
        Record.objects.all().filter(count_date__month='05'),
        Record.objects.all().filter(count_date__month='06'),
        Record.objects.all().filter(count_date__month='07'),
        Record.objects.all().filter(count_date__month='08'),
        Record.objects.all().filter(count_date__month='09'),
        Record.objects.all().filter(count_date__month='10'),
        Record.objects.all().filter(count_date__month='11'),
        Record.objects.all().filter(count_date__month='12'),
    ]
    
    #record_month_db_list의 원소들로 부터 월별 count를 계산
    for cur_month in range(0, 12):
        cur_month_db_list = record_month_db_list[cur_month]

        total_count = 0
        for day_db in cur_month_db_list:
            total_count += day_db.get_count_and_month()['all_count']
        record_month_list[cur_month]['all_count'] = total_count
        record_month_list[cur_month]['month'] = str(cur_month + 1)

    #return 오브젝트
    return_object = {
        "record_day_list": record_day_list[-5:],
        "record_month_list": record_month_list
    }

    return render(request, 'statistics.html', return_object)


def summary(request):
    return render(request, 'summary.html')


def analysis(request):
    if request.method == "POST":
        today = datetime.today()
        current_year = today.year
        current_month = today.month
        current_day = today.day
        standard_time = time(0, 0, 0)
        time_list = []
        time_list2 = []
        check_point = Record.objects.filter(count_date__year=current_year, count_date__month=current_month,
                                            count_date__day=current_day).count()
        if check_point < 4:
            all_count = TodayTraffic.objects.filter(date=today.date()).count()
            for i in range(0, 23, 1):
                time_list.insert(i, (TodayTraffic.objects.filter(date=today.date(), time__gte=standard_time) & TodayTraffic.objects.filter(date=today.date(), time__lte=standard_time.replace(hour=i + 1))).count())
                standard_time = standard_time.replace(hour=i+1)
            time_list.insert(23, (TodayTraffic.objects.filter(date=today.date(), time__gte='23:00:00') & TodayTraffic.objects.filter(date=today.date(), time__lte='00:00:00')).count())
            Record.objects.create(all_count=all_count,
                                  time_1=time_list[0], time_2=time_list[1], time_3=time_list[2], time_4=time_list[3],
                                  time_5=time_list[4], time_6=time_list[5], time_7=time_list[6], time_8=time_list[7],
                                  time_9=time_list[8], time_10=time_list[9], time_11=time_list[10], time_12=time_list[11],
                                  time_13=time_list[12], time_14=time_list[13], time_15=time_list[14], time_16=time_list[15],
                                  time_17=time_list[16], time_18=time_list[17], time_19=time_list[18], time_20=time_list[19],
                                  time_21=time_list[20], time_22=time_list[21], time_23=time_list[22], time_24=time_list[23],)
            TodayTraffic.objects.all().delete()
            return redirect('analysis')
        else:
            record_list = Record.objects.all()
            return render(request, 'analysis.html', {"record_list": record_list, 'error_message': "Error", })
    else:
        record_list = Record.objects.all()
        record_list = [record.get_values() for record in record_list]
        return render(request, 'analysis.html', {"record_list": record_list})
