# Generated by Django 4.2.5 on 2023-11-19 07:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Niwi_Travels', '0016_alter_travelpackage_status'),
    ]

    operations = [
        migrations.AddField(
            model_name='passenger',
            name='status',
            field=models.CharField(default='Pending', max_length=20),
        ),
    ]
