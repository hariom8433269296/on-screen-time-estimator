from flask import Flask, render_template, redirect, url_for, flash
from flask_wtf.csrf import CSRFProtect
from forms import VideoForm, ImageForm
from werkzeug.utils import secure_filename
from datetime import datetime
import image_proc
import on_screen

import os
import threading

app = Flask(__name__)

csrf = CSRFProtect(app)
app.config['SECRET_KEY'] = 'secret_key'

def process_video(video):
    try:
        video_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
        video.save(os.path.join('uploads_video', video_filename))
        image_proc.main()
        on_screen.func()
    except Exception as e:
        print(f"Error processing video: {e}")


@app.route('/')
def home():
    # Check if there's any result available to display
    # You may need to modify this part according to your requirements
    result = None  # Placeholder for the result
    return render_template('home.html', result=result)

@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    form = VideoForm()

    if form.validate_on_submit():
        video = form.video.data
        # Start a new thread to process the video asynchronously
        threading.Thread(target=process_video, args=(video,)).start()
        flash('Video uploaded successfully! Processing...', 'success')

    return render_template('upload_video.html', form=form)

@app.route('/upload_images', methods=['GET', 'POST'])
def upload_images():
    form = ImageForm()

    if form.validate_on_submit():
        for i in range(1, 6):
            image = getattr(form, f'image{i}').data
            if image:
                image_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}.png"
                image.save(os.path.join('uploads_images', image_filename))

        flash('Images uploaded successfully!', 'success')

    return render_template('upload_images.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
