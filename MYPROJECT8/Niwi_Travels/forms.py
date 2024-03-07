from django import forms



class CustomFileInput(forms.ClearableFileInput):
    def get_context(self, name, value, attrs):
        accept = "application/pdf, image/jpeg, image/jpg, image/png"
        attrs["accept"] = accept
        context = super().get_context(name, value, attrs)
        return context
    
    
class PassengerInfoForm(forms.Form):
    passenger_name = forms.CharField(
        max_length=100,
        label='Passenger Name',
        widget=forms.TextInput(attrs={'placeholder': 'Passenger Name'})
    )
    passenger_age = forms.IntegerField(
        label='Passenger Age',
        widget=forms.TextInput(attrs={'placeholder': 'Passenger Age'})
    )
    proof_of_id = forms.FileField(
        label='Proof of ID',
        help_text='Upload a PDF file',
        widget=CustomFileInput(attrs={'placeholder': 'Upload a PDF file'})
    )



    # You can add more fields as needed for your application.
# forms.py

# forms.py
from django import forms
from django.forms import inlineformset_factory
from .models import CustomPackage, Day

class TravelPackageForm(forms.ModelForm):
    # Add this field for package image
    package_image = forms.ImageField(label='Package Image', required=False)

    class Meta:
        model = CustomPackage
        fields = ['category', 'name', 'description', 'days', 'nights', 'package_image','price']
class DayForm(forms.ModelForm):
    class Meta:
        model = Day
        fields = ['day_number', 'image', 'image_description']

DayFormSet = inlineformset_factory(CustomPackage, Day, form=DayForm, extra=1)

class CustomPackageForm(forms.ModelForm):
    class Meta:
        model = CustomPackage
        fields = ['category', 'name', 'description', 'days', 'nights', 'price', 'package_image','status']

        widgets = {
            'description': forms.Textarea(attrs={'rows': 4}),
        }

# forms.py


# forms.py
from django import forms

class SearchForm(forms.Form):
    search_query = forms.CharField(label='Search for a place', max_length=100, widget=forms.TextInput(attrs={'id': 'autocomplete-input'}))

