from flask import Flask,flash, render_template,make_response,send_file,jsonify, request, redirect, url_for,Response, stream_with_context
from flask.wrappers import Request
from werkzeug.utils import secure_filename
import train as train
import os
import json
import tensorflow as tf
import random
import zipfile
import zipfile
import logging
import threading, queue,time
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ____ init ____
q = queue.Queue()
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
ALLOWED_EXTENSIONS = {'zip'}
# ____ processing function ____

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
      if model_training == "mobilenet":
         result = model_created.mobileNet('mobilenet')
         return result
      elif model_training == "resnet50":
         result = model_created.resnet50()
         return result
      elif model_training == "mobilenetv2":
         result = model_created.mobileNet('mobilenetv2')
         return result
      elif model_training == "envgame_leaf_disease":
         # model_created.img_height = 224
         # model_created.img_width = 224
         result = model_created.envgame_leaf_disease()
         return result
      else:
         return "result"
   # except:
   #    print("====================")
   #    print("Wrong directory structure")
   #    print("====================")
   #    return "<b style='color:red'>Wrong directory structure or System error </b>"
      
# ____ router flask ____

def worker():
   global result_gl
   while True:
         data = q.get()
         print("================ SYSTEM LOG ========================")
         print(f'Working on {data["data_dir"]}')
         print("====================================================")
         result_gl = traning(
            data_dir= data['data_dir']
               ,img_height = data['img_height']
               ,img_width= data['img_width']
               ,batch_size= data['batch_size']
               ,epoch = data['epoch']
               ,name_model= data['name_model']
               ,model_training = data['model_training']
            )
         print("================ SYSTEM LOG ========================")
         print(f'Finished {data["data_dir"]}')
         print("====================================================")
         q.task_done()

# turn-on the worker thread
threading.Thread(target=worker, daemon=True).start()
print('All task requests sent\n', end='')
# block until all tasks are done

@app.route("/",methods = ["GET","POST"])
def index():
   if(request.method=="POST"):
      try:
         data= request.form.get("data")
         data = json.loads(data)
         uploaded_file = request.files['file']
         if uploaded_file and allowed_file(uploaded_file.filename):
            path_save = "uploads/" + uploaded_file.filename
            uploaded_file.save(path_save)
            print("================ SYSTEM LOG ========================")
            print('GET FILE *.zip')
            print("====================================================")
            data_progress(path_save)
            os.remove(path_save)
            print("================ SYSTEM LOG ========================")
            print('Extract oke !!! *.zip')
            print("====================================================")
            task_info = {
                 'data_dir': path_save.replace(".zip","")
               ,'img_height' : data['img_height']
               ,'img_width': data['img_width']
               ,'batch_size': data['batch_size']
               ,'epoch' : data['epoch']
               ,'name_model': data['name_model']
               ,'model_training' : data['model_training']
            }
            q.put(task_info)
            # result = traning(
            # data_dir= path_save.replace(".zip","")
            #    ,img_height = data['img_height']
            #    ,img_width= data['img_width']
            #    ,batch_size= data['batch_size']
            #    ,epoch = data['epoch']
            #    ,name_model= data['name_model'] 
            #    ,model_training = data['model_training']
            #    )
         # return redirect("historys/", code=302)
         return render_template("loading.html", jsonify(json.dumps(result_gl)))
      except:
         return render_template("upload.html")
   if(request.method=="GET"):
      return render_template("upload.html")

@app.route('/models')
def models():
   files = os.listdir('models')
   return str(files)
def get_val():
   try:
      return result_gl
   except:
      return ''

@app.route('/static/log/nohup.out')
def nohup():
   with open('static/log/nohup.out') as f:
      lines = f.read() 
      return json.dumps({
         "nohup": lines,
         "result": get_val()
         })

@app.route('/download/<name_model>', methods=['GET', 'POST'])
def download(name_model):
   print(name_model)
   return send_file("models/"+name_model, as_attachment=True)

@app.route('/historys/<history_model>', methods=['GET', 'POST'])
def historys(history_model):
   try:
      with open('historys/'+ history_model + ".txt") as f:
         lines = f.read()
         lines = lines.replace("'", '"')
         rs = json.loads(lines)
         return render_template("history.html", history = rs)
   except:
      return render_template("error.html", name = history_model) 
   

q.join()
print('All work completed')

# ____ system config ____

if __name__ == '__main__':
   app.run(debug=True)