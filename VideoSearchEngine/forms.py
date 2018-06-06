# forms.py
 
from wtforms import Form, StringField, SelectField
 
class VideoSearchForm(Form):
    search = StringField('')

class AddVideoForm(Form):
    name = StringField('name')
    url = StringField('url')
    