# Generated by Django 4.2.5 on 2023-11-14 04:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Niwi_Travels', '0011_booking_passenger_limit'),
    ]

    operations = [
        migrations.AlterField(
            model_name='booking',
            name='updated_at',
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]
