import datetime
import sqlite3

import django
from django.db import models
from django.conf import settings
from django.utils import timezone


# Create your models here.

# class TodayTraffic(models.Model):
#     person_id = models.IntegerField(primary_key=True, unique=True)
#     date = models.DateField(auto_now_add=True, blank=True)
#     time = models.TimeField(auto_now_add=True, blank=True)
#
#     def publish(self):
#         self.save()
#
#     def insertData(self, person_id):
#         conn = sqlite3.connect("./db.sqlite3")
#         cur = conn.cursor()
#         query = '''INSERT INTO detectme_traffic (person_id, date, time)
#                     VALUES (?, ?, ?)'''
#         cur.execute(query, (person_id, timezone.now(), "10:00"))
#         conn.commit()
#         cur.close()
#         conn.close()
#라인 정보
class line_list(models.Model):
    line_number = models.IntegerField(null=False)
    line_one_x = models.IntegerField(null=False)
    line_one_y = models.IntegerField(null=False)
    line_two_x = models.IntegerField(null=False)
    line_two_y = models.IntegerField(null=False)

class TodayTraffic(models.Model):
    person_id = models.IntegerField(default=-1, primary_key=True)
    date = models.DateField()
    time = models.TimeField()

    def get_person_id(self):
        return {'person_id': self.person_id}

    def get_month(self):
        return self.date.strftime("%m")


#오늘 기록
class TodayRecord(models.Model):
    all_count = models.IntegerField(default=0)
    time_1 = models.IntegerField(null=True, default=0)  # 0~1
    time_2 = models.IntegerField(null=True, default=0)  # 1~2
    time_3 = models.IntegerField(null=True, default=0)  # 2~3
    time_4 = models.IntegerField(null=True, default=0)  # 3~4
    time_5 = models.IntegerField(null=True, default=0)  # 4~5
    time_6 = models.IntegerField(null=True, default=0)  # 5~6
    time_7 = models.IntegerField(null=True, default=0)  # 6~7
    time_8 = models.IntegerField(null=True, default=0)  # 7~8
    time_9 = models.IntegerField(null=True, default=0)  # 8~9
    time_10 = models.IntegerField(null=True, default=0)  # 9~10
    time_11 = models.IntegerField(null=True, default=0)  # 10~11
    time_12 = models.IntegerField(null=True, default=0)  # 11~12
    time_13 = models.IntegerField(null=True, default=0)  # 12~13
    time_14 = models.IntegerField(null=True, default=0)  # 13~14
    time_15 = models.IntegerField(null=True, default=0)  # 14~15
    time_16 = models.IntegerField(null=True, default=0)  # 15~16
    time_17 = models.IntegerField(null=True, default=0)  # 16~17
    time_18 = models.IntegerField(null=True, default=0)  # 17~18
    time_19 = models.IntegerField(null=True, default=0)  # 18~19
    time_20 = models.IntegerField(null=True, default=0)  # 19~20
    time_21 = models.IntegerField(null=True, default=0)  # 20~21
    time_22 = models.IntegerField(null=True, default=0)  # 21~22
    time_23 = models.IntegerField(null=True, default=0)  # 22~23
    time_24 = models.IntegerField(null=True, default=0)  # 23~24
    count_date = models.DateTimeField(default=timezone.now())

    def get_values(self):
        return {'all_count': self.all_count,
                'time_1': self.time_1,   'time_2': self.time_2,   'time_3': self.time_3,   'time_4' : self.time_4,
                'time_5': self.time_5,   'time_6': self.time_6,   'time_7': self.time_7,   'time_8' : self.time_8,
                'time_9': self.time_9,   'time_10': self.time_10, 'time_11': self.time_11, 'time_12' : self.time_12,
                'time_13': self.time_13, 'time_14': self.time_14, 'time_15': self.time_15, 'time_16' : self.time_16,
                'time_17': self.time_17, 'time_18': self.time_18, 'time_19': self.time_19, 'time_20' : self.time_20,
                'time_21': self.time_21, 'time_22': self.time_22, 'time_23': self.time_23, 'time_24' : self.time_24 }



#과거 기록들
class Record(models.Model):
    all_count = models.IntegerField(default=0)
    time_1 = models.IntegerField(null=True, default=0)  # 0~1
    time_2 = models.IntegerField(null=True, default=0)  # 1~2
    time_3 = models.IntegerField(null=True, default=0)  # 2~3
    time_4 = models.IntegerField(null=True, default=0)  # 3~4
    time_5 = models.IntegerField(null=True, default=0)  # 4~5
    time_6 = models.IntegerField(null=True, default=0)  # 5~6
    time_7 = models.IntegerField(null=True, default=0)  # 6~7
    time_8 = models.IntegerField(null=True, default=0)  # 7~8
    time_9 = models.IntegerField(null=True, default=0)  # 8~9
    time_10 = models.IntegerField(null=True, default=0)  # 9~10
    time_11 = models.IntegerField(null=True, default=0)  # 10~11
    time_12 = models.IntegerField(null=True, default=0)  # 11~12
    time_13 = models.IntegerField(null=True, default=0)  # 12~13
    time_14 = models.IntegerField(null=True, default=0)  # 13~14
    time_15 = models.IntegerField(null=True, default=0)  # 14~15
    time_16 = models.IntegerField(null=True, default=0)  # 15~16
    time_17 = models.IntegerField(null=True, default=0)  # 16~17
    time_18 = models.IntegerField(null=True, default=0)  # 17~18
    time_19 = models.IntegerField(null=True, default=0)  # 18~19
    time_20 = models.IntegerField(null=True, default=0)  # 19~20
    time_21 = models.IntegerField(null=True, default=0)  # 20~21
    time_22 = models.IntegerField(null=True, default=0)  # 21~22
    time_23 = models.IntegerField(null=True, default=0)  # 22~23
    time_24 = models.IntegerField(null=True, default=0)  # 23~24
    count_date = models.DateTimeField(null=True, default=timezone.now())

    def get_values(self):
        return {'all_count': self.all_count,
                'time_1': self.time_1,   'time_2': self.time_2,   'time_3': self.time_3,   'time_4' : self.time_4,
                'time_5': self.time_5,   'time_6': self.time_6,   'time_7': self.time_7,   'time_8' : self.time_8,
                'time_9': self.time_9,   'time_10': self.time_10, 'time_11': self.time_11, 'time_12' : self.time_12,
                'time_13': self.time_13, 'time_14': self.time_14, 'time_15': self.time_15, 'time_16' : self.time_16,
                'time_17': self.time_17, 'time_18': self.time_18, 'time_19': self.time_19, 'time_20' : self.time_20,
                'time_21': self.time_21, 'time_22': self.time_22, 'time_23': self.time_23, 'time_24' : self.time_24}

    def get_count_and_month(self):
        return {'all_count': self.all_count, 'month': self.count_date.strftime("%m"),
                'day': self.count_date.strftime("%d")}


    #def __str__(self):
     #   return self.click_date
