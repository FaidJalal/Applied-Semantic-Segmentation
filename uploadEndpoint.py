import os
from flask import Flask, request, redirect, url_for,jsonify
from werkzeug.utils import secure_filename
import json
import predict
from flask_cors import CORS

UPLOAD_FOLDER = '/home/aatirah/Desktop/NewPROJ/upload'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            #flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            result = predict.pred(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return jsonify(result)
            #return json.dumps({'status':'success'}) 
    return

if __name__ == '__main__':
    app.run(debug=True)
