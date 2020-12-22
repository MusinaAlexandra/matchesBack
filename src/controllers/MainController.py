from datetime import datetime

from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/footbal')
def hello_world():
    week = request.args.get('week')
    win_kind = {
        'firstWin' : 1,
        'draw' : 2,
        'firstLoos' : 3,
    }
    games = [
        {
            'date': datetime.now().strftime("%H:%M:%S"),
            'firstPlayer': {
                'id': 1,
                'name': 'Renn',
            },
            'secondPlayer': {
                'id': 2,
                'name': 'Chelsi',
            },
            'result' : {
                'win': 7.35,
                'lose': 89.10,
                'draw': 3.55,
            },
            'portfolio' : {
                'firstName' : 'Krasnodar',
                'secondName': 'Sevilia',
                'kind': win_kind['draw']
            },
        },
        {
            'date': datetime.now().strftime("%H:%M:%S"),
            'firstPlayer': {
                'id': 3,
                'name': 'Zenit',
            },
            'secondPlayer': {
                'id': 4,
                'name': 'Chelsi',
            },
            'result': {
                'win': 7.35,
                'lose': 89.10,
                'draw': 3.55,
            },
            'portfolio': {
                'firstName': 'Ural',
                'secondName': 'Sevilia',
                'kind': win_kind['firstWin']
            },
        },

    ]
    return jsonify(games)