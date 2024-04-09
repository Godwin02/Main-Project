# custom_filters.py
import random
from django import template

register = template.Library()

@register.filter(name='get_value_from_key')
def get_value_from_key(dictionary, key):
    return dictionary.get(key, 0)

@register.filter(name='random_color')
def random_color(package):
    # Generate a random color in hexadecimal format
    return f'#{random.randint(0, 0xFFFFFF):06x}'