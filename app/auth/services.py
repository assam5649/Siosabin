from werkzeug.security import generate_password_hash, check_password_hash
from .models import User
from . import db

def register_service(data):
    hashed_password = generate_password_hash(data['password'], method='sha256')
    new_user = User(username=data['username'], password_hash=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    return {'id': new_user.id, 'username': new_user.username}

def login_service(data):
    user = User.query.filter_by(username=data['username']).first()
    if user and check_password_hash(user.password_hash, data['password']):
        return {'message': 'Login successful'}
    return {'message': 'Invalid credentials'}, 401