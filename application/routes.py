from flask import render_template, url_for, flash, redirect, request
from application import app, db, bcrypt
from application.forms import RegistrationForm, LoginForm
from application.database import User, Wallet, Stock
from flask_login import login_user, current_user, logout_user, login_required
from application.data import AlphaVantageAPI
from application.graph import GraphBuilder
from application.model import ArimaModelBuilder
from application.model import ArimaModelBuilder, LSTMModelBuilder
import pandas as pd
import json
import plotly
import plotly.express as px

av = AlphaVantageAPI()
graph = GraphBuilder()
arima = ArimaModelBuilder()
ltsm = LSTMModelBuilder()






@app.route("/")
def index():
    return render_template('index.html')



@app.route("/home")
def home():
    stocks = Stock.query.all()
    return render_template('home.html', title='Home', stocks=stocks)




@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)



@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route("/wallet")
@login_required
def wallet():
    list_stocks = []
    wallets = Wallet.query.filter_by(user_id=current_user.id).all()
    for wallet in wallets:
        stock = Stock.query.filter_by(id = wallet.stock_id).first()
        list_stocks.append(stock)

    return render_template('wallet.html', title='Wallet', list_stocks=list_stocks)


@app.route("/addwallet/<string:symbol>")
@login_required
def addwallet(symbol):
    user = User.query.filter_by(id=current_user.id).first()
    stock = Stock.query.filter_by(symbol=symbol).first()
    if Wallet.query.filter_by(user_id=user.id, stock_id=stock.id).first():
        flash('Already added', 'danger')
    else:
        wallet = Wallet(user_id=user.id, stock_id=stock.id)
        db.session.add(wallet)
        db.session.commit()
        flash('Added to wallet', 'success')
    return render_template('addwallet.html', stock=stock)

@app.route('/search', methods=['GET','POST'])
def search():
    query = request.args.get('search')
    stocks = Stock.query.filter((Stock.name.like('%' + query + '%')) | (Stock.symbol.like('%' + query + '%'))).all()
    return render_template('search.html', title='Search', stocks=stocks, query=query)


@app.route("/stock/<string:symbol>")
def stock(symbol):
    stock = Stock.query.filter_by(symbol=symbol).first()

    Description = av.get_overview(symbol).get('Description')

    graph.get_data(symbol)
    close = graph.close_graph()
    graph1JSON = json.dumps(close, cls=plotly.utils.PlotlyJSONEncoder)

    volume = graph.volume_graph()
    graph2JSON = json.dumps(volume, cls=plotly.utils.PlotlyJSONEncoder)

    m_avg = graph.moving_average()
    graph3JSON = json.dumps(m_avg, cls=plotly.utils.PlotlyJSONEncoder)

    daily_return = graph.daily_return()
    graph4JSON = json.dumps(daily_return, cls=plotly.utils.PlotlyJSONEncoder)

    risk = graph.risk()
    graph5JSON = json.dumps(risk, cls=plotly.utils.PlotlyJSONEncoder)

    arima.get_data(symbol)
    arima.parameters()
    arima.split_data()
    predict = arima.predict()
    graph6JSON = json.dumps(predict, cls=plotly.utils.PlotlyJSONEncoder)

    forecast = arima.forecast()
    graph7JSON = json.dumps(forecast, cls=plotly.utils.PlotlyJSONEncoder)

    ltsm.get_data(symbol)
    ltsm.prepare_data()
    ltsm.split_data()
    ltsm_graph = ltsm.graph()
    graph8JSON = json.dumps(ltsm_graph, cls=plotly.utils.PlotlyJSONEncoder)



    return render_template('stock.html',title='Stock', symbol=symbol, stock=stock, Description=Description,
                           close=graph1JSON, volume=graph2JSON,
                           m_avg=graph3JSON, daily_return=graph4JSON,
                           risk=graph5JSON, predict=graph6JSON,
                           forecast=graph7JSON,ltsm_graph=graph8JSON)


@app.errorhandler(404)
def error_404(error):
    return render_template('404.html'), 404



@app.errorhandler(500)
def error_500(error):
    return render_template('500.html'), 500



