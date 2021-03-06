# Generated by Django 3.2.5 on 2022-04-26 02:17

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Record',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('all_count', models.IntegerField(default=0)),
                ('time_1', models.IntegerField(default=0, null=True)),
                ('time_2', models.IntegerField(default=0, null=True)),
                ('time_3', models.IntegerField(default=0, null=True)),
                ('time_4', models.IntegerField(default=0, null=True)),
                ('time_5', models.IntegerField(default=0, null=True)),
                ('time_6', models.IntegerField(default=0, null=True)),
                ('time_7', models.IntegerField(default=0, null=True)),
                ('time_8', models.IntegerField(default=0, null=True)),
                ('time_9', models.IntegerField(default=0, null=True)),
                ('time_10', models.IntegerField(default=0, null=True)),
                ('time_11', models.IntegerField(default=0, null=True)),
                ('time_12', models.IntegerField(default=0, null=True)),
                ('time_13', models.IntegerField(default=0, null=True)),
                ('time_14', models.IntegerField(default=0, null=True)),
                ('time_15', models.IntegerField(default=0, null=True)),
                ('time_16', models.IntegerField(default=0, null=True)),
                ('time_17', models.IntegerField(default=0, null=True)),
                ('time_18', models.IntegerField(default=0, null=True)),
                ('time_19', models.IntegerField(default=0, null=True)),
                ('time_20', models.IntegerField(default=0, null=True)),
                ('time_21', models.IntegerField(default=0, null=True)),
                ('time_22', models.IntegerField(default=0, null=True)),
                ('time_23', models.IntegerField(default=0, null=True)),
                ('time_24', models.IntegerField(default=0, null=True)),
                ('count_date', models.DateTimeField(default=datetime.datetime(2022, 4, 26, 2, 17, 27, 82929))),
            ],
        ),
        migrations.CreateModel(
            name='TodayRecord',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('all_count', models.IntegerField(default=0)),
                ('time_1', models.IntegerField(default=0, null=True)),
                ('time_2', models.IntegerField(default=0, null=True)),
                ('time_3', models.IntegerField(default=0, null=True)),
                ('time_4', models.IntegerField(default=0, null=True)),
                ('time_5', models.IntegerField(default=0, null=True)),
                ('time_6', models.IntegerField(default=0, null=True)),
                ('time_7', models.IntegerField(default=0, null=True)),
                ('time_8', models.IntegerField(default=0, null=True)),
                ('time_9', models.IntegerField(default=0, null=True)),
                ('time_10', models.IntegerField(default=0, null=True)),
                ('time_11', models.IntegerField(default=0, null=True)),
                ('time_12', models.IntegerField(default=0, null=True)),
                ('time_13', models.IntegerField(default=0, null=True)),
                ('time_14', models.IntegerField(default=0, null=True)),
                ('time_15', models.IntegerField(default=0, null=True)),
                ('time_16', models.IntegerField(default=0, null=True)),
                ('time_17', models.IntegerField(default=0, null=True)),
                ('time_18', models.IntegerField(default=0, null=True)),
                ('time_19', models.IntegerField(default=0, null=True)),
                ('time_20', models.IntegerField(default=0, null=True)),
                ('time_21', models.IntegerField(default=0, null=True)),
                ('time_22', models.IntegerField(default=0, null=True)),
                ('time_23', models.IntegerField(default=0, null=True)),
                ('time_24', models.IntegerField(default=0, null=True)),
                ('count_date', models.DateTimeField(default=datetime.datetime(2022, 4, 26, 2, 17, 27, 81930))),
            ],
        ),
        migrations.CreateModel(
            name='TodayTraffic',
            fields=[
                ('person_id', models.IntegerField(default=-1, primary_key=True, serialize=False)),
                ('date', models.DateField()),
                ('time', models.TimeField()),
            ],
        ),
    ]
