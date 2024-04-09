# Generated by Django 4.2.5 on 2024-03-15 10:19

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('Niwi_Travels', '0035_customrating'),
    ]

    operations = [
        migrations.CreateModel(
            name='CustomPayment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('is_paid', models.BooleanField(default=False)),
                ('razor_pay_order_id', models.CharField(blank=True, max_length=100, null=True)),
                ('razor_pay_payment_id', models.CharField(blank=True, max_length=100, null=True)),
                ('razor_pay_payment_signature', models.CharField(blank=True, max_length=100, null=True)),
                ('amount', models.DecimalField(decimal_places=2, max_digits=10)),
                ('payment_date', models.DateTimeField(auto_now_add=True)),
                ('booking', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='Niwi_Travels.custombooking')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]