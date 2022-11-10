import sqlite3
import getpass


class MLDatabase(object):

    def __init__(self, dbname: str) -> None:
        self.conn = sqlite3.connect(dbname)
        self.sql = '''
        CREATE TABLE IF NOT EXISTS `text_classification` (
            `username` varchar(64) NOT NULL,
            `create_time` datetime DEFAULT CURRENT_TIMESTAMP,
            `model` varchar(64) NOT NULL,
            `dataset` varchar(64) NOT NULL,
            `hypermeters` varchar(1024),
            `metrics` varchar(512),
            `accuracy` float
        );
        '''
        self.insert_sql = '''
        INSERT INTO `text_classification` (username, model, dataset, hypermeters, metrics, accuracy)
        VALUES ("{}", "{}", "{}", "{}", "{}", {})
        '''
        self.username = getpass.getuser()
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute(self.sql)
        self.conn.commit()

    def insert_experiment_result(self, result: dict):
        formatted_sql = self.insert_sql.format(self.username, result['model'], result['dataset'], result['hypermeters'], result['metrics'], result['accuracy'])
        cursor = self.conn.cursor()
        cursor.execute(formatted_sql)
        self.conn.commit()

    def close(self):
        self.conn.close()


if __name__ == '__main__':
    dbname = './database/ml.db'
    database = MLDatabase(dbname)
    result = {
        'model': 'TextCNN',
        'hypermeters': 'lr=1e-5, num_filters=256',
        'metrics': 'f1=0.94, acc=0.96',
        'accuracy': 0.946312
    }
    database.insert_experiment_result(result)
