from flask import Flask
from flask_cors import CORS
from flask import request,jsonify

from dotenv import load_dotenv
from email_validator import validate_email, EmailNotValidError
from Pages.extension import db,bcrypt,migrate,jwt_manager
from flask_jwt_extended import create_access_token
from datetime import timedelta
from Model.User import User

import os


load_dotenv()

app = Flask(__name__)

CORS(app, origins=["http://localhost:3000"])

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")



db.init_app(app)
bcrypt.init_app(app)
migrate.init_app(app,db)
jwt_manager.init_app(app)
@app.post("/createuser")
def createuser():
    body=request.get_json() 

    email=body["email"]
    address=body["address"]
    password=body["password"]
    name=body["name"]
    if not email or not password:
       return jsonify({"response":"email or password is missing "}),400
    
    if not validEmail(email):
       return jsonify({"response":"not a valid email"}),400
    
    hash_password=User.hashpassword(password)

    newuser=User(name,email,address,hash_password)

    db.session.add(newuser)
    db.session.commit()

    return jsonify({
       "response":"created a new user succefully"
    }),201

@app.post("/login")
def login():
   body=request.get_json()
   myemail=body["email"]
   mypassword=body["password"]

   Myuser=User.query.filter_by(email=myemail).first()
   
   if not Myuser:
      return jsonify({"response":"wrong email"}),400
   get_hash_password=Myuser.password_hash

   if not User.checkhashpassword(get_hash_password,mypassword):
      return jsonify({"response":"password is incorect"}),400
   
   myjwttoken=create_access_token(
      identity={"userId":Myuser.id,"email":Myuser.email},
      expires_delta=timedelta(minutes=1)
      
      )

   return jsonify({"response":"correct credential hence login successfully!!","token":myjwttoken}),200
   


def validEmail(email):
   try:
      validate_email(email)
      return True
   except EmailNotValidError:
      return False