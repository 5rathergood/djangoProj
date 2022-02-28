import sqlite3

import django
from django.db import models
from django.conf import settings
from django.utils import timezone


# Create your models here.

class Traffic(models.Model):
    person_id = models.IntegerField(default=-1)
    create_time = models.DateTimeField(default=django.utils.timezone.now)
    delete_time = models.DateTimeField(null=True)

    def publish(self):
        self.create_time = timezone.now()
        self.save()

    def insertData(self, person_id):
        conn = sqlite3.connect("./db.sqlite3")
        cur = conn.cursor()
        query = '''INSERT INTO detectme_traffic (person_id, create_time, delete_time)
                    VALUES (?, ?, ?)'''
        cur.execute(query, (person_id, timezone.now(), timezone.now()))
        conn.commit()
        cur.close()
        conn.close()


    #def __str__(self):
     #   return self.click_date
