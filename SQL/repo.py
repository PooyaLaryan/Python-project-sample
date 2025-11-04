from SQL.transaction import SQLTransaction

class Repository:
    def __init__(self):
        self.t = SQLTransaction()
        pass

    def h3_to_index(self):
        df = self.t.execute_query('''
SELECT ROW_NUMBER() OVER(ORDER BY T.H3Id) AS RowNumber, T.H3Id AS h3 FROM
 (SELECT DISTINCT h.H3Id FROM ML.dbo.H3 h) AS T

        ''')
        return df


    