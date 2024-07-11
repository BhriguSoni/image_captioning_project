# image_captioning_app/views.py

import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from keras.models import Model
import pickle
from gensim.models import Word2Vec
import cv2
from .forms import ImageUploadForm

max_length=35 

model_path = os.path.join(os.path.dirname(__file__), "model")

model = load_model(os.path.join(model_path, "model.keras"))

with open(os.path.join(model_path, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)


with open(os.path.join(model_path, 'word_index.pkl'), 'rb') as f:
    word_index = pickle.load(f)

feature_extractor = Xception(weights="imagenet", include_top=False)
x = feature_extractor.output
x = GlobalAveragePooling2D()(x)
feature_extractor = Model(inputs=feature_extractor.input, outputs=x)

def idx_to_word(integer, word_index):
    for word, index in word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, word_index, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = [word_index.get(word, 0) for word in in_text.split()]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, word_index)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.split()
    final_caption = final_caption[1:-1]  
    final_caption = ' '.join(final_caption)
    return final_caption

def generate_caption(image_name):
    image_id = image_name.split('.')[0]
    img_path = os.path.join(settings.MEDIA_ROOT, image_name)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image_id not in features:
        
        img = cv2.resize(image, (299, 299))  
        img = np.expand_dims(img, axis=0)  
        img = preprocess_input(img)  
        feature_vector = feature_extractor.predict(img)  
    else:
        feature_vector = features[image_id]
    
    y_pred = predict_caption(model, feature_vector, word_index, max_length)
    return y_pred


def index(request):
    caption = None
    uploaded_file_url = None
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid() and 'image' in request.FILES:
            image = form.cleaned_data["image"]
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            uploaded_file_url = fs.url(filename)
            try:
                caption = generate_caption(filename)
            except Exception as e:
                caption = f"Error generating caption: {str(e)}"
        else:
            form.add_error('image', 'Image upload is required.')
    else:
        form = ImageUploadForm()
    return render(request, "index.html", {
        "form": form,
        "uploaded_file_url": uploaded_file_url,
        "caption": caption
    })