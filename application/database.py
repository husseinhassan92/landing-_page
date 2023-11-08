from application import db, login_manager
from flask_login import UserMixin

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



class Wallet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    stock_id = db.Column(db.Integer, db.ForeignKey('stock.id'))

    def __repr__(self):
        return f"Wallet('{self.user_id}', '{self.stock_id}')"


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    wallet = db.relationship('Wallet', backref='user', lazy=True)


    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"

class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(20))
    last_sale = db.Column(db.Integer)
    net_change = db.Column(db.Integer)
    market_cap = db.Column(db.Integer)
    ipo_year = db.Column(db.Integer)
    volume = db.Column(db.Integer)
    country = db.Column(db.String(120))
    sector = db.Column(db.String(120))
    industry = db.Column(db.String(120))
    wallet = db.relationship('Wallet', backref='stock', lazy=True)

    def __repr__(self):
        return f"Stock('{self.symbol}', '{self.name}')"