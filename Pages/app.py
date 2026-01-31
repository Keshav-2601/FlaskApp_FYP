from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask import Request,Response,jsonify
from dotenv import load_dotenv
import os


load_dotenv()

app = Flask(__name__)

CORS(app, origins=["http://localhost:3000"])

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

@app.get("/createuser")
def createuser():
    body=jsonify(Request)
    print(body)
    
    return "Flask is running"

@app.get("/login")
def login(req,res):
    return "login is correct"