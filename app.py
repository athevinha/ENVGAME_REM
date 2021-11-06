from flask import Flask,flash, render_template,make_response,send_file,jsonify, request, redirect, url_for,Response, stream_with_context
from flask.wrappers import Request
from werkzeug.utils import secure_filename
import train as train
import os
import json
import tensorflow as tf
import random
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

def traning(data_dir,img_height,img_width,batch_size,name_model,epoch,model_training):
   # try:
      model_created = train.create_model(
         data_dir = data_dir
         ,img_height =img_height
         ,img_width = img_width
         ,batch_size = batch_size
         ,name_model = name_model
         ,epoch = epoch
         ,model_training = model_training)
      result = model_created.mobileNet()
      return result
   # except:
   #    print("====================")
   #    print("Wrong directory structure")
   #    print("====================")
   #    return "<b style='color:red'>Wrong directory structure or System error </b>"
      


@app.route("/",methods = ["GET","POST"])
def index():
   if(request.method=="POST"):
      data= request.form.get("data")
      data = json.loads(data)
      uploaded_file = request.files['file']
      if uploaded_file and allowed_file(uploaded_file.filename):
         path_save = "uploads/" + uploaded_file.filename
         uploaded_file.save(path_save)
         data_progress(path_save)
         os.remove(path_save)

         result = traning(
           data_dir= path_save.replace(".zip","")
            ,img_height = data['img_height']
            ,img_width= data['img_width']
            ,batch_size= data['batch_size']
            ,epoch = data['epoch']
            ,name_model= data['name_model']
            ,model_training = data['model_training']
            )

      # return render_template("loading.html", jsonify(json.dumps(result)))
      return jsonify(json.dumps(result))
   if(request.method=="GET"):
      return render_template("upload.html")

   
@app.route('/download/<name_model>', methods=['GET', 'POST'])
def download(name_model):
   print(name_model)
   return send_file("models/"+name_model, as_attachment=True)


@app.route('/stream')
def streamed_response():
    def generate():
        yield 'Hello '
        yield request.args['name']
        yield '!'
    return app.response_class(stream_with_context(generate()))
if __name__ == '__main__':
   app.run(debug = True)