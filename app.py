import pickle
 
from flask import Flask, request, render_template, redirect
from collections import Counter

import cv2
import os
import numpy as np
 
app=Flask(__name__)
app.config['upload_folder']="uploads"

 #loding the model
with open("color_knn.pkl",'rb') as f:
          model=pickle.load(f)

def p(im):
   img=cv2.imread(im)
   if img is not None:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = cv2.resize(img,(50,50))
      img = img.reshape(-1,3)
      output = model.predict(img)
      counts = Counter(output)
      op=[]
      for color, count in counts.items():
         if count > 100:
            op.append(str(color))
            print(color,count)
      return op
   return None
 
@app.route('/',methods=['GET','POST'])
def default():
   return render_template("home.html")
 


@app.route('/predict',methods=['POST']) 
def predict():
   img=request.files['image']  
   path=os.path.join(app.config['upload_folder'],'upload.jpg')
    
   img.save(path)
   i=p(path)
    
   if i is not None:
      return render_template("result.html",post=i)
   return redirect('/')  


if __name__=="__main__":
   app.run(debug=True)
