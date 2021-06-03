import sqlite3


class Properties():
    def __init__(self, db_name="database/db.db"):
        connection = sqlite3.connect(db_name)
        cursor = connection.cursor()

        cursor.execute('SELECT value FROM properties WHERE name = "num_steps";')
        self.__num_steps = cursor.fetchone()[0]

        cursor.execute('SELECT value FROM properties WHERE name = "style_weight";')
        self.__style_weight = cursor.fetchone()[0]

        cursor.execute('SELECT value FROM properties WHERE name = "content_weight";')
        self.__content_weight = cursor.fetchone()[0]

        if self.__num_steps == None or self.__style_weight == None or self.__content_weight == None:
            raise RuntimeError("there is no necessary properties") 
        
        connection.close()
           

    @property
    def num_steps(self):
        return self.__num_steps
    @property
    def style_weight(self):
        return self.__style_weight
    @property
    def content_weight(self):
        return self.__content_weight
