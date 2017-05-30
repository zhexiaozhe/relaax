from pandas import read_csv


class DataLoader(object):
    """ Loads data as pandas sub data-frame for each player """
    def __init__(self, data_path, fill_nan=None):
        df = read_csv(data_path, header=0)
        if fill_nan is not None:
            df.fillna(value=fill_nan, inplace=True)

        # get a list of all players IDs
        players_id = df['sr_player_id'].unique().tolist()
        self.players_data = []

        for player in players_id:
            self.players_data.append(df.loc[df.sr_player_id == player])

        # alias for simplicity
        self.pd = self.players_data
