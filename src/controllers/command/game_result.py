class game_result:
    def __init__(self, draw, lose, win):
        self.draw = draw  # ничья
        self.lose = lose  # потеря first_player
        self.win = win    # выигрыш first_player