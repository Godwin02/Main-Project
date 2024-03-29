# Generated by Django 4.2.5 on 2024-02-18 05:09

import datetime
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('Niwi_Travels', '0025_alter_custompackage_package_image'),
    ]

    operations = [
        migrations.AddField(
            model_name='custompackage',
            name='status',
            field=models.CharField(choices=[('Post', 'Post'), ('Save', 'Save')], default='Save', max_length=10),
        ),
        migrations.CreateModel(
            name='CustomPassenger',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('passenger_name', models.CharField(max_length=100)),
                ('passenger_age', models.PositiveIntegerField()),
                ('proof_of_id', models.FileField(upload_to='passenger_ids/')),
                ('status', models.CharField(default='Pending', max_length=20)),
                ('package', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='Niwi_Travels.travelpackage')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='CustomBooking',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('status', models.CharField(choices=[('Pending', 'Pending'), ('Confirmed', 'Confirmed'), ('Cancelled', 'Cancelled')], default='Pending', max_length=20)),
                ('boarding', models.DateField(default=datetime.date(2024, 2, 18))),
                ('start_date', models.DateTimeField(auto_now_add=True)),
                ('passenger_limit', models.IntegerField(default=0)),
                ('children', models.IntegerField(default=0)),
                ('package', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='Niwi_Travels.custompackage')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
