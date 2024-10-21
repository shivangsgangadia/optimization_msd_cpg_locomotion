import sqlite3
from typing import List

import pandas
import GeneticAlgorithm


class DataStorage:
    def __init__(self, database_file_name: str):
        self.database_file_name = database_file_name
        self.database_connection = sqlite3.connect(database_file_name)
        self.cursor = self.database_connection.cursor()

    def store_with_command(self, command: str):
        try:
            self.cursor.execute(command)
            self.database_connection.commit()
        except sqlite3.OperationalError as e:
            print(e)
            print(command)

    def get_pandas_dataframe(self):
        return pandas.read_sql_query('SELECT * FROM simulation;', self.database_connection)

    def close(self):
        self.database_connection.close()

    @staticmethod
    def store_to_csv(data: List[float]):
        csv_file = open("amplitude_data_95.csv", 'a')
        csv_file.write(str(data).strip('[]') + '\n')
        csv_file.close()
