from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SubmitField

class VideoForm(FlaskForm):
    video = FileField('Upload Video', validators=[FileAllowed(['mp4', 'mkv', 'avi'])])
    submit_video = SubmitField('Upload Video')

class ImageForm(FlaskForm):
    image1 = FileField('Upload Image 1', validators=[FileAllowed(['jpg', 'jpeg', 'png'])])
    image2 = FileField('Upload Image 2', validators=[FileAllowed(['jpg', 'jpeg', 'png'])])
    image3 = FileField('Upload Image 3', validators=[FileAllowed(['jpg', 'jpeg', 'png'])])
    image4 = FileField('Upload Image 4', validators=[FileAllowed(['jpg', 'jpeg', 'png'])])
    image5 = FileField('Upload Image 5', validators=[FileAllowed(['jpg', 'jpeg', 'png'])])
    submit_images = SubmitField('Upload Images')
