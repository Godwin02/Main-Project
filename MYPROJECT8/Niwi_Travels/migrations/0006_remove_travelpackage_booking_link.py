# Generated by Django 4.2.5 on 2023-10-31 15:16

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('Niwi_Travels', '0005_remove_travelpackage_ratings'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='travelpackage',
            name='booking_link',
        ),
    ]