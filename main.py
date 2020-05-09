from flask import Flask,render_template,request
import numpy as np
app = Flask(__name__)
import pickle
file=open('model.pkl','rb')
model=pickle.load(file)
file.close()


@app.route('/', methods=["GET","POST"])
def hello_world():
    '''
    iq=[1,2,3,4,5,6,7,8,9,0,12,12,13]
    oq=model.predict(np.array([iq]))
    return 'Hello, Worlds!'+str(oq)
'''
    if request.method =="POST":
        md=request.form
        CRIM=int(md['CRIM'])
        ZN=int(md['ZN'])
        INDUS=int(md['INDUS'])
        RS=int(md['RS'])
        NOX=int(md['NOX'])
        RM=int(md['RM'])
        AGE=int(md['AGE'])
        DIS=int(md['DIS'])
        HD=int(md['HD'])
        TAX=int(md['TAX'])
        PTRATIO=int(md['PTRATIO'])
        B=int(md['B'])
        LSTAT=int(md['LSTAT'])
        ip=[CRIM,ZN,INDUS,RS,NOX,RM,AGE,DIS,HD,TAX,PTRATIO,B,LSTAT]
        ip_pred=model.predict(np.array([ip]))
        return render_template('show.html',price=ip_pred[0])

    return render_template('index.html')
    
    
if __name__ == "__main__":
    app.run(debug=True)