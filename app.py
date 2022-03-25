from re import I
from flask import Flask,flash, render_template,make_response,send_file,jsonify, request, redirect, url_for,Response, stream_with_context
from flask.wrappers import Request
from werkzeug.utils import secure_filename
import train as train
import os
import json
import tensorflow as tf
import random
import zipfile
from PIL import Image
import logging
import numpy as np
import threading, queue,time
import urllib.request
from flask_cors import CORS, cross_origin


app = Flask(__name__)
q = queue.Queue()
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
ALLOWED_EXTENSIONS = {'zip','png','jpg'}
zip_files = 0



def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
      #  print(dirs)
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))
      
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def filter_space(string):
   # print(string)
   return ' '.join(string.split())

def folder_structure(startpath):
   structure_dist = {"data":[]}
   for root, dirs, files in os.walk(startpath):
      files.sort()
      # print(dirs)
      # print(files)
      structure_dist['data'].append({
         "classes": root,
         "files": files
      })
   return json.dumps(structure_dist)

def data_progress(data_dir):
   global zip_files
   zip_files = os.path.getsize(data_dir) / 10000000
   print('app trainend',zip_files)
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
         ,model_training = model_training
         ,zip_files = zip_files
         )
      if model_training == "mobilenet":
         result = model_created.mobileNet('mobilenet')
         return result
      elif model_training == "resnet50":
         result = model_created.resnet50()
      elif model_training == "inceptionV3":
         result = model_created.inceptionV3()
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
         print("<br/>================ SYSTEM LOG ========================")
         print(f'WORK ON {data["name_model"]}')
         print("====================================================<br/>")
         result_gl = traning(
            data_dir= data['data_dir']
               ,img_height = data['img_height']
               ,img_width= data['img_width']
               ,batch_size= data['batch_size']
               ,epoch = data['epoch']
               ,name_model= data['name_model']
               ,model_training = data['model_training']
            )
         print("<br/>================ SYSTEM LOG ========================")
         print(f'FINISHED {data["name_model"]}')
         print("====================================================<br/>")
         
         q.task_done()

threading.Thread(target=worker, daemon=True).start()
print('All task requests sent\n', end='')

@app.route("/",methods = ["GET","POST"])
@cross_origin()
def index():
   if(request.method=="POST"):
      try:
         data= request.form.get("data")
         data = json.loads(data)
         uploaded_file = request.files['file']
         if uploaded_file and allowed_file(uploaded_file.filename):
            path_save = "uploads/" + uploaded_file.filename
            uploaded_file.save(path_save)
            # print("================ SYSTEM LOG ========================")
            # print('GET FILE *.zip')
            # print("====================================================")
            data_progress(path_save)
            os.remove(path_save)
            print("<br/>================ SYSTEM LOG ========================")
            print('Extract oke !!! *.zip')
            print('Your model is in queue !!! *.zip')
            print("==================================================== <br/>")
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

@app.route('/getHistoryFES/<history_model>', methods=['GET', 'POST'])
@cross_origin()
def getHistoryFES(history_model):
   try:
      with open('historys/'+ history_model + ".txt") as f:
         lines = f.read()
         lines = lines.replace("'", '"')
         rs = json.loads(lines)
         return json.dumps({
            "history":rs,
            "status": 200,
            })
   except:
      # print('download')
      return {
         "error": "error",
         "status":404,
         }
      
@app.route('/downloadDataFes/<name_folder>', methods=['GET', 'POST'])
@cross_origin()
def downloadDataFes(name_folder):
   zipf = zipfile.ZipFile('zip/'+ name_folder+'.zip', 'w', zipfile.ZIP_DEFLATED)
   zipdir('static/'+ name_folder +"/", zipf)
   zipf.close()
   return send_file('zip/'+ name_folder+'.zip', as_attachment=True)

@app.route('/getAllClasses/<name_folder>', methods=['GET', 'POST'])
@cross_origin()
def getAllClasses(name_folder):
   dir = 'static/' + name_folder
   classesd = os.listdir(dir)
   return json.dumps({ 
      'classes': classesd,
   })

@app.route('/showDatasStructureFes/<name_folder>', methods=['GET', 'POST'])
@cross_origin()
def showDatasFes(name_folder):
   results = folder_structure('static/'+ name_folder + "/")
   return results


dir = 'static/addData'
roots = []
for root, dirs, files in os.walk(dir):
   if root != dir:
      roots.append(root)

@app.route('/showLeafDataFes/<name_folder>', methods=['GET', 'POST'])
@cross_origin()
def showLeafDataFes(name_folder):
   classes = roots[random.randint(0, len(roots) - 1)]
   files = os.listdir(classes)
   name = files[random.randint(0, len(files) - 1)]
   url = '/' + classes +'/'+ name
   return json.dumps({ 
      'url':url,
      'classes':classes.replace('static/' + name_folder+'/',''),
      'name': name
   })

@app.route('/pushDataFes/', methods=['GET', 'POST'])
@cross_origin()
def pushLeafDataFes():
   url = request.args.get('url')
   classes= request.args.get('classes')
   name = request.args.get('name')
   url = url.replace(" ", "%20")
   opener = urllib.request.build_opener()
   opener.addheaders = [('User-Agent', 'MyApp/1.0')]
   urllib.request.install_opener(opener)
   urllib.request.urlretrieve(url, "static/exampleData/"+ classes + "/" + name)
   return json.dumps({ 
      'url':url,
      'classes':classes.replace('static/addData/',''),
      'name': name
   }) 


@app.route("/pushImageFES",methods = ["GET","POST"])
@cross_origin()
def pushImageFES():
   if(request.method=="POST"):
      try:
         classes = request.form.get("classes")
         print(classes)
         # data = json.loads(data)
         uploaded_file = request.files['file']
         # uploaded_file = uploaded_file.resize((256,256))
         print(uploaded_file)
         if uploaded_file and allowed_file(uploaded_file.filename):
            path_save = "static/exampleData/"+ classes + "/00___USER_DATA___00_" + uploaded_file.filename
            uploaded_file.save(path_save)
         return "Push image '"+ uploaded_file.filename + "' to '" + classes +"' success!"
      except Exception as e :
         print(e)
         return "error"
   if(request.method=="GET"):
      return render_template("upload.html")


def data_augmentation_random_brightness(image,label = "ok",seed=(1,0)):
  aug_image = tf.image.adjust_brightness(image, delta = random.uniform(0.5,0.6))
  return aug_image

def data_augmentation_random_contrast(image,label = "ok",seed=(2,0)):
  aug_image = tf.image.adjust_contrast(image, random.uniform(0.95,1))
  return aug_image

def data_augmentation_random_gamma(image,label = "ok",seed=(2,0)):
  aug_image = tf.image.adjust_gamma(image, random.uniform(0.5,0.7))
  return aug_image

def data_augmentation_random_rotation(image,label = "ok",central_fraction=(0.5)):
  data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2)])
  aug_image = data_augmentation(image)
#   aug_image = tf.image.resize(image, [256,256])
  return aug_image

@app.route('/augDataFes/', methods=['GET', 'POST'])
@cross_origin()
def augDataFes():
   name = request.args.get('name')
   classes= request.args.get('classes')
   augs = request.args.get('aug').split(',')
   path = "static/exampleData/" + classes + "/" + name
   im = Image.open(path)
   for aug in augs:
      im = tf.keras.preprocessing.image.img_to_array(im)
      if aug =='g':
         im = data_augmentation_random_gamma(image = im)
      if aug == 'b':
         im = data_augmentation_random_brightness(image = im)
      if aug == 'c':
         im = data_augmentation_random_contrast(image = im)
      if aug == 'r':
         im = data_augmentation_random_rotation(image = im)
   tf.keras.utils.save_img(path , im, data_format=None, file_format=None, scale=True)
   print(im)
   return json.dumps({ 
      'img':str(im),
      'classes':classes.replace('static/addData/',''),
      'name': name
   })

   # return send_file("static/"+name_folder, as_attachment=True)
@app.route('/cleanLog',methods = ["GET","POST"])
def cleanLog():
   print('--- CLEAN LOG ---')
   f = open("static/log/nohup.out", "r+")
   f.truncate(0)
   return f.read()


q.join()
print('All work completed')

# ____ system config ____

if __name__ == '__main__':
   app.run(host='0.0.0.0',port=5000,debug=True)