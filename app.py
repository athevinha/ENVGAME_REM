from flask import Flask,flash, render_template, request, redirect, url_for,Response, stream_with_context
from werkzeug.utils import secure_filename
import train as train
import os
import tensorflow as tf
import zipfile
app = Flask(__name__)
import zipfile

ALLOWED_EXTENSIONS = {'zip'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def data_progress(data_dir):
   with zipfile.ZipFile(data_dir, 'r') as zip_ref:
    zip_ref.extractall("uploads/")

def traning(data_dir,img_height,img_width,batch_size):
   try:
      model_created = train.create_model(data_dir,img_height,img_width,name_model="flower")
      model_created.mobileNet()
   except:
      print("====================")
      print("Wrong directory structure")
      print("====================")


@app.route("/",methods = ["GET","POST"])
def index():
   if(request.method=="POST"):
      uploaded_file = request.files['file']
      if uploaded_file and allowed_file(uploaded_file.filename):
         path_save = "uploads/" + uploaded_file.filename
         uploaded_file.save(path_save)
         data_progress(path_save)
         os.remove(path_save)
         traning(path_save.replace(".zip",""),224, 224, 64)
      return render_template("upload.html")
   if(request.method=="GET"):
      return render_template("upload.html")
      
if __name__ == '__main__':
   app.run(debug = True)