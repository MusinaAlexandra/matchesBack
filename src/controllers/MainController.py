from flask import Flask, jsonify, request
from flaskext.mysql import MySQL

app = Flask(__name__)

mysql = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'root'
app.config['MYSQL_DATABASE_DB'] = 'matches'
app.config['MYSQL_DATABASE_HOST'] = 'mysql-reports.docker'
mysql.init_app(app)

win_kind = {
    'firstWin': 1,
    'draw': 2,
    'firstLoos': 3,
}


@app.route('/footbal')
def get_games():
    week = request.args.get('week')
    win_kind = {
        'firstWin': 1,
        'draw': 2,
        'firstLoos': 3,
    }

    conn = mysql.connect()
    cursor = conn.cursor()

    sql = "select * FROM games g where weekofyear(g.game_date) = " + week;
    cursor.execute(sql)
    games_data = cursor.fetchall()

    result = []
    for game in games_data:
        result.append({
            'date': game[1],
            'first_player': game[2],
            'second_player': game[3],
            'kind': getKind(game[4])
        })

    return jsonify(result)


def getKind(game_result):
    count = game_result.split("-")
    if (count[0] > count[1]):
        return win_kind['firstWin']
    if (count[0] < count[1]):
        return win_kind['firstLoos']

    return win_kind['draw']
