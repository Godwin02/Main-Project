# Generated by Django 4.2.5 on 2023-10-09 10:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Niwi_Travels', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='is_traveller',
            field=models.BooleanField(default=True),
        ),
    ]