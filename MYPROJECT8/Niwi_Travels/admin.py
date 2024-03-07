from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, Traveller, Driver

# Customize the UserAdmin to display your custom fields
class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'is_traveller', 'is_driver',)
    def get_queryset(self, request):
        # Only include users who are not staff members
        return super().get_queryset(request).exclude(is_staff=True)


class TravellerAdmin(admin.ModelAdmin):
    list_display = ('user', 'first_name', 'phone_number', 'country', 'gender')

class DriverAdmin(admin.ModelAdmin):
    list_display = ('user', 'first_name','contact_email', 'contact_phone_number', 'verification')

# Register the custom admin classes with the respective models
admin.site.register(Traveller, TravellerAdmin)
admin.site.register(Driver, DriverAdmin)

# Register the User model with the custom admin class
admin.site.register(User, CustomUserAdmin)

# Register the NormalUser and CollegeUser models
# admin.site.register(Traveller)
# admin.site.register(Driver)