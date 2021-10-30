from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
import zipfile
app = Flask(__name__)
import zipfile
@app.route("/",methods = ["GET","POST"])
def index():
   if(request.method=="POST"):
      uploaded_file = request.files['file']
      if uploaded_file.filename != '':
         path_save = "uploads/" + uploaded_file.filename
         uploaded_file.save(path_save)
         data_progress(path_save)
         os.remove(path_save)
         traning(path_save.replace(".zip",""),240, 240, 64)
         
      return render_template("upload.html")
   if(request.method=="GET"):
      return render_template("upload.html")

def data_progress(data_dir):
   with zipfile.ZipFile(data_dir, 'r') as zip_ref:
    zip_ref.extractall("uploads/")



def traning(data_dir,img_height,img_width,batch_size):
   train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  shuffle=False)


    