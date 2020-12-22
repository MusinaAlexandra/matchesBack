from flask import Flask, jsonify, request
from flaskext.mysql import MySQL

from Predictor import predict

app = Flask(__name__)

mysql = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'root'
app.config['MYSQL_DATABASE_DB'] = 'matches'
app.config['MYSQL_DATABASE_HOST'] = 'mysql-reports.docker'
mysql.init_app(app)


@app.route('/football')
def get_football_data():
    week = request.args.get('week')

    games = get_games(week)
    result = predict(games)

    return jsonify(result)


def get_games(week):
    games_data = load_games(week)

    games = []
    for game in games_data:
        games.append({
            'date': game[1],
            'first_player': game[2],
            'second_player': game[3],
            'round': game[4]
        })

    return games


def load_games(week):
    conn = mysql.connect()
    cursor = conn.cursor()

    sql = "select * FROM games g where weekofyear(g.game_date) = " + week
    cursor.execute(sql)
    games_data = cursor.fetchall()

    return games_data
