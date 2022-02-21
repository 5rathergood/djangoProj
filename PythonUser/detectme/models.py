from django.db import models
from django.conf import settings
from django.utils import timezone


# Create your models here.

class TestDB(models.Model):
    click_date = models.DateTimeField(default=timezone.now(), primary_key=True)
    n_click = models.IntegerField(default=-1)

    def publish(self):
        self.click_date = timezone.now()
        self.save()

    #def __str__(self):
     #   return self.click_date
