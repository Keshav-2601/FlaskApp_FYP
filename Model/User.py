from flask import Flask
from Pages.extension import bcrypt,db
class User(db.Model): # inherit from db.model db is the obj got from sqlAlchemy in flask
    __tablename__ = "users" 
    id = db.Column(db.Integer, primary_key=True,autoincrement=True)
    name=db.Column(db.String(120),nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    address=db.Column(db.String(120),unique=False,nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)

    def __init__(self,name,email,address,password): ## parametrized constructor here. self is like the this  
        self.name=name
        self.email=email
        self.address=address
        self.password_hash=password

    @staticmethod
    def hashpassword(password):
         password_hash=bcrypt.generate_password_hash(password).decode("utf-8")
         return password_hash
    
    @staticmethod
    def checkhashpassword(password_hash,password):
       return bcrypt.check_password_hash(password_hash,password)
    
