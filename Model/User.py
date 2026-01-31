from flask import Flask
from Pages import app 
from app import db ## coming from app or main page.
from flask_bcrypt import Bcrypt
class User(db.Model): # inherit from db.model db is the obj got from sqlAlchemy in flask
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True,autoincrement=True)
    name = db.Column(db.String(100), nullable=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    address=db.Column(db.String(120),unique=False,nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)

    def __init__(self,email,address,password):  ## parametrized constructor here. self is like the this 
        self.email=email
        self.address=address
        self.password=password


    def hashpassowrd(password):
        bcrypt = Bcrypt(app)
        password_hash=bcrypt.generate_password_hash(password)
        
        return password_hash
    
