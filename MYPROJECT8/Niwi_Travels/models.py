from datetime import timezone
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.core.validators import MinValueValidator, MaxValueValidator

class User(AbstractUser): 
    is_traveller = models.BooleanField(default=True)
    is_driver = models.BooleanField(default=False)
    
    def _str_(self):
         return f"{self.username} ({'Traveller' if self.is_traveller else 'Driver'})"

class Traveller(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True, related_name='traveller')
    first_name = models.CharField(max_length=30,null=True)
    last_name = models.CharField(max_length=30,null=True)
    phone_number = models.CharField(max_length=15, null=True)
    country = models.CharField(max_length=100, null=True)
    gender = models.CharField(max_length=10, null=True)
    profile_photo = models.ImageField(blank=True, null=True, upload_to='traveller_profile_photos/')

    def _str_(self):
        return self.user.username
    
class Driver(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True,related_name='driver')
    first_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    contact_email = models.EmailField(max_length=254)
    contact_phone_number = models.CharField(max_length=15)
    profile_photo = models.ImageField(blank=True, null=True, upload_to='driver_profile_photos/')
    verification = models.CharField(max_length=20, default='Pending')
    admin_notes = models.TextField(blank=True, null=True)
    license = models.FileField(blank=True, null=True, upload_to='college_pdf_copies/')
    date_of_birth = models.DateField(null=True, blank=True)  # Add date_of_birth field
    location = models.CharField(max_length=100, null=True, blank=True)  # Add location field

    def _str_(self):
        return self.user.username
    




class TravelPackage(models.Model):
    STATUS_CHOICES = (
        ('Running', 'Running'),
        ('Paused', 'Paused'),
    )
    ACCOMMODATION_CHOICES = (
        ('Included', 'Included'),
        ('Excluded', 'Excluded'),
    )
    FEED_CHOICES = (
        ('Save', 'Save'),
        ('Post', 'Post'),
    )

    package_name = models.CharField(max_length=100)
    description = models.TextField()
    destination = models.CharField(max_length=100)
    duration = models.PositiveIntegerField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    start_date = models.DateField()
    end_date = models.DateField()
    accommodation = models.CharField(max_length=100, choices=ACCOMMODATION_CHOICES,default='Included')
    # meals = models.CharField(max_length=100)
    transportation = models.CharField(max_length=100)
    activities = models.TextField()
    inclusions = models.TextField()
    exclusions = models.TextField()
    images = models.ImageField(upload_to='package_images/')
    # ratings = models.FloatField()
    availability = models.PositiveIntegerField()
    booking_deadline = models.DateField()
    category = models.CharField(max_length=50)
    tags = models.CharField(max_length=100)
    cancellation_policy = models.TextField()
    # booking_link = models.URLField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Pending')
    feed = models.CharField(max_length=10, choices=FEED_CHOICES, default='Save')


    def __str__(self):
        return self.package_name

class PackageImage(models.Model):
    package = models.ForeignKey(TravelPackage, on_delete=models.CASCADE)  # Assuming TravelPackage is your package model
    image = models.ImageField(upload_to='package_images/')

    def __str__(self):
        return f"Image for {self.package.package_name}"
    


    


from django.utils import timezone
class Booking(models.Model):
    STATUS_CHOICES = (
        ('Pending', 'Pending'),
        ('Confirmed', 'Confirmed'),
        ('Cancelled', 'Cancelled'),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    package = models.ForeignKey(TravelPackage, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    passenger_limit = models.IntegerField(default=1)  # Adjust default value as needed
    children=models.IntegerField(default=0)
    
    # Fields related to payment
    # total_price = models.DecimalField(max_digits=10, decimal_places=2)
    # payment_method = models.CharField(max_length=50)
    # payment_status = models.CharField(max_length=20, choices=PAYMENT_STATUS_CHOICES, default='Pending')
    # payment_date = models.DateTimeField(blank=True, null=True)
    
    # Add other common fields for your Booking model

    def __str__(self):
        return f'Booking ID: {self.id}, User: {self.user}, Package: {self.package}, Status: {self.status}'

class Passenger(models.Model):
    passenger_name = models.CharField(max_length=100)
    passenger_age = models.PositiveIntegerField()
    proof_of_id = models.FileField(upload_to='passenger_ids/')
    package = models.ForeignKey(TravelPackage, on_delete=models.CASCADE)  # Assuming TravelPackage is your package model
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Associate each passenger with a user
    status = models.CharField(max_length=20, default='Pending')  # Assuming 'Pending' is the default status
    def __str__(self):
        return self.passenger_name
    


@receiver(post_save, sender=Booking)
def update_passengers_on_booking_update(sender, instance, **kwargs):
    # Update Passenger instances related to the updated Booking instance
    Passenger.objects.filter(package=instance.package, user=instance.user).update(status=instance.status)
    
class Payment(models.Model):
    booking=models.ForeignKey(Booking, on_delete=models.CASCADE, blank=True, null=True)
    is_paid = models.BooleanField(default=False)
    razor_pay_order_id=models.CharField(max_length=100, blank=True, null=True)
    razor_pay_payment_id=models.CharField(max_length=100, blank=True, null=True)
    razor_pay_payment_signature=models.CharField(max_length=100, blank=True, null=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Link the payment to a customer
    amount = models.DecimalField(max_digits=10, decimal_places=2)  # Store the payment amount
    payment_date = models.DateTimeField(auto_now_add=True)  # Date and time of the payment
    # Add other fields as per your requirements, like payment status, transaction ID, etc.
    
    def __str__(self):
        return f"Payment of {self.amount} by {self.customer.username} on {self.payment_date}"
    

class Rating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    package = models.ForeignKey(TravelPackage, on_delete=models.CASCADE)
    stars = models.IntegerField(null=False, blank=False, default=0)
    description = models.TextField(blank=True, null=True)


# models.py

from django.db import models

class Activity(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField()
    custom_package = models.ForeignKey('CustomPackage', on_delete=models.CASCADE)

    def __str__(self):
        return self.name

class Day(models.Model):
    custom_package = models.ForeignKey('CustomPackage', on_delete=models.CASCADE)
    day_number = models.PositiveIntegerField()
    image = models.ImageField(upload_to='day_images/', blank=True, null=True)
    image_description = models.TextField()

    def __str__(self):
        return f'Day {self.day_number} - {self.custom_package.name}'

class CustomPackage(models.Model):
    CATEGORY_CHOICES = [
        ('Honeymoon', 'Honeymoon'),
        ('Adventure', 'Adventure'),
        ('Family', 'Family'),
    ]
    STATUS_CHOICES = [
        ('Post', 'Post'),
        ('Save', 'Save'),
    ]
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    name = models.CharField(max_length=255)
    description = models.TextField()
    days = models.PositiveIntegerField()
    nights = models.PositiveIntegerField()
    package_image = models.ImageField(upload_to='custom_package_images/', null=True, blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2,null=True)  # Add this line for the price field
    status = models.CharField(max_length=10, choices=STATUS_CHOICES,)  # Add the new field


    def __str__(self):
        return self.name



class CustomBooking(models.Model):
    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Confirmed', 'Confirmed'),
        ('Cancelled', 'Cancelled'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    package = models.ForeignKey('CustomPackage', on_delete=models.CASCADE, related_name='custom_bookings')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Pending')
    boarding = models.CharField(max_length=50, blank=True, null=True)
    start_date = models.DateField(null=True)
    passenger_limit = models.IntegerField(default=1, validators=[MinValueValidator(1)], null=True)
    children = models.IntegerField(default=0, null=True)

    def __str__(self):
        return f'Booking ID: {self.id}, User: {self.user}, Package: {self.package}, Status: {self.status}, Boarding: {self.boarding}'

class CustomPassenger(models.Model):
    passenger_name = models.CharField(max_length=100)
    passenger_age = models.PositiveIntegerField()
    proof_of_id = models.FileField(upload_to='passenger_ids/')
    package = models.ForeignKey(CustomPackage, on_delete=models.CASCADE, related_name='custom_passengers')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, default='Pending')

    booking = models.ForeignKey(CustomBooking, on_delete=models.CASCADE, related_name='passengers')

    def __str__(self):
        return f'Passenger ID: {self.id}, Name: {self.passenger_name}, Age: {self.passenger_age}'
    

# models.py
from django.db import models

class Place(models.Model):
    name = models.CharField(max_length=255)
    latitude = models.FloatField()
    longitude = models.FloatField()

    def __str__(self):
        return self.name
