import datetime
import sqlite3

import django
from django.db import models
from django.conf import settings
from django.utils import timezone


# Create your models here.

# class Traffic(models.Model):
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


class Traffic(models.Model):
    person_id = models.IntegerField(default=-1)
    date = models.DateField()
    time = models.TimeField()


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
    count_date = models.DateTimeField(default=timezone.now())


    #def __str__(self):
     #   return self.click_date
