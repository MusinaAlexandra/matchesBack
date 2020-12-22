# matchesBack

крутится на http://localhost:5000/


Как установит фласк: 

    https://flask.palletsprojects.com/en/1.1.x/installation/

Как запускать:

    надо нактить скрипты для базы из папки /db (сначала db.sql, потом остальные) на MySql

    

в MyController.py подставить свои данные для коннекта к базе :

    app.config['MYSQL_DATABASE_USER'] = 'root'
    app.config['MYSQL_DATABASE_PASSWORD'] = 'root'
    app.config['MYSQL_DATABASE_DB'] = 'matches'
    app.config['MYSQL_DATABASE_HOST'] = 'mysql-reports.docker'


Непостредственно запуск:

    export FLASK_APP=controllers/MainController.py
    export FLASK_ENV=development
    flask run

Если что смотреть тут: https://flask.palletsprojects.com/en/1.1.x/quickstart/#a-minimal-application
