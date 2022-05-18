from datetime import date, time, datetime


from django.shortcuts import render, redirect
from django.urls import reverse
from django.views import View
from django.views.decorators import gzip
from django.http import StreamingHttpResponse, JsonResponse, HttpRequest
from django.db import IntegrityError
import threading
import cv2

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

import queue
import ObjectTrack as OT
import time

ot_q = queue.Queue()
line_q = queue.Queue()
OT_thread = threading.Thread(target=OT.ObjectTrack,args=(ot_q, line_q))
#OT_thread.start()

Cam_Alive = False
line_check = False

class VideoCamera(object):
    def __init__(self):
        self.frame = cv2.imread('static/ds/prepare.jpg')
        global Cam_Alive
        #print('init: ', Cam_Alive)
        
        #이전에 카메라 객체를 생성한 경우
        if Cam_Alive:
            Cam_Alive = False
            
            #기존 쓰레드가 종료될 때까지 1초 기다림
            time.sleep(1)
        #print('working')

        Cam_Alive = True
        self.update_thread = threading.Thread(target=self.update, args=())
        self.update_thread.start()
        if not OT_thread.is_alive():
            line_q.put(True)
            OT_thread.start()

    def __del__(self):
        print('cam delete')

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        global Cam_Alive, line_check
        self.frame = ot_q.get()
        while Cam_Alive:
            self.frame = ot_q.get()
            if line_check:
                line_q.put(True)
                line_check = False


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



def home(request):
    #최초 실행시 today_record_list가 비어있다면 default row를 하나 생성
    today_record_list = TodayRecord.objects.all()
    if len(today_record_list) == 0:
        first_low = TodayRecord.objects.create()
        first_low.save()

    traffic_list = TodayTraffic.objects.all()

    traffic_list = [traffic.get_person_id() for traffic in traffic_list]
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


def InitTodayTraffic(request):
    TodayTraffic.objects.all().delete()

    data = {
        "state": "success"
    }
    return JsonResponse(data)


def InitTodayRecord(request):
    TodayRecord.objects.all().delete()
    first_low = TodayRecord.objects.create()
    first_low.save()

    data = {
        "state": "success"
    }

    return JsonResponse(data)



def summary(request):
    return render(request, 'summary.html')


# def analysis(request):
#     if request.method == "POST":
#         today = datetime.today()
#         current_year = today.year
#         current_month = today.month
#         current_day = today.day
#         standard_time = time(0, 0, 0)
#         time_list = []
#         time_list2 = []
#         check_point = Record.objects.filter(count_date__year=current_year, count_date__month=current_month,
#                                             count_date__day=current_day).count()
#         if check_point < 4:
#             all_count = TodayTraffic.objects.filter(date=today.date()).count()
#             for i in range(0, 23, 1):
#                 time_list.insert(i, (TodayTraffic.objects.filter(date=today.date(), time__gte=standard_time) & TodayTraffic.objects.filter(date=today.date(), time__lte=standard_time.replace(hour=i + 1))).count())
#                 standard_time = standard_time.replace(hour=i+1)
#             time_list.insert(23, (TodayTraffic.objects.filter(date=today.date(), time__gte='23:00:00') & TodayTraffic.objects.filter(date=today.date(), time__lte='00:00:00')).count())
#             Record.objects.create(all_count=all_count,
#                                   time_1=time_list[0], time_2=time_list[1], time_3=time_list[2], time_4=time_list[3],
#                                   time_5=time_list[4], time_6=time_list[5], time_7=time_list[6], time_8=time_list[7],
#                                   time_9=time_list[8], time_10=time_list[9], time_11=time_list[10], time_12=time_list[11],
#                                   time_13=time_list[12], time_14=time_list[13], time_15=time_list[14], time_16=time_list[15],
#                                   time_17=time_list[16], time_18=time_list[17], time_19=time_list[18], time_20=time_list[19],
#                                   time_21=time_list[20], time_22=time_list[21], time_23=time_list[22], time_24=time_list[23],)
#             TodayTraffic.objects.all().delete()
#             return redirect('analysis')
#         else:
#             record_list = Record.objects.all()
#             return render(request, 'analysis.html', {"record_list": record_list, 'error_message': "Error", })
#     else:
#         record_list = Record.objects.all()
#         record_list = [record.get_values() for record in record_list]
#         return render(request, 'analysis.html', {"record_list": record_list})

class AnalysisCreateView(View):
    '''
    2022.05.05
    박병제
    분석 생성 만듬.
    문제 1.유동인구 db연결
        2.도로명 주소를 통해 찾기
    '''
    def get(self, request : HttpRequest, *args, **kwargs):
        context = {}
        context['attraction'] = "먼저 면적을 입력하세요"
        return render(request,'analysis.html',context)
    def post(self, request : HttpRequest, *args, **kwargs):
        context = {}

        area = int(request.POST['area'])
        population = 10 # 유동인구
        total_attraction = 2.9917 #상수 값임 get_cur_shop_attraction() 함수를 사용할 때 쓰여야함
        distance = 100 #거리값은 변경되어야함 수동으로 입력
        #calc_attraction
        calc_attraction = area/(pow(distance, 2))

        #get_cur_shop_attraction
        get_cur_shop_attraction = calc_attraction/(calc_attraction + total_attraction)
        attraction = get_cur_shop_attraction * population
        context['attraction'] = str(round(attraction * 100, 3)) + "%"
        return render(request, 'analysis.html',context)

class writelineView(View):
    def get(self, request:HttpRequest, *args,**kwargs):
        line_q.put(True)
        return redirect('..')

