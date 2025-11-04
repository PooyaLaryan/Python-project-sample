import pyodbc
import pandas as pd

class SQLTransaction:
    def __init__(self):
        # تنظیمات اتصال به دیتابیس
        database = 'master'  # نام دیتابیس

        server = '172.31.56.41'  # یا IP سرور
        username = 'cs_lariyan'  # نام کاربری
        password = 'ERGY4bK31U7Qd2PF6T?R_W'  # رمز عبور

        #server = '172.31.56.40'  # یا IP سرور
        # username = 'CS_Khorshidi'  # نام کاربری
        # password = 'nz6Lr2gxzVJr'  # رمز عبور

        driver = '{ODBC Driver 17 for SQL Server}'  # درایور ODBC

        # ساخت رشته اتصال
        self.connection_string = f'''
            DRIVER={driver};
            SERVER={server};
            DATABASE={database};
            UID={username};
            PWD={password};
            TrustServerCertificate=yes;'''

    def execute_query(self, query) -> pd.DataFrame:
        try:
            self.connection = pyodbc.connect(self.connection_string)
            cursor = self.connection.cursor()
            cursor.execute(query)

            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchall()

            df = pd.DataFrame.from_records(rows, columns=columns)

            cursor.close()
            self.connection.close()

            return df

        except Exception as e:
            print("خطا در اتصال یا اجرای کوئری:", e)

    def execute_sp(self, sp, params: dict = None) -> pd.DataFrame:
        try:
            self.connection = pyodbc.connect(self.connection_string)
            cursor = self.connection.cursor()

            values = []
            sql_code = f"EXEC {sp} "
            if params != None:
                for item in params:
                    sql_code += f"@{item}=?,"
                    values.append(params[item])

                sql_code = sql_code[:-1]

            print(sql_code)
            print(values)
            cursor.execute(sql_code, values)

            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchall()

            df = pd.DataFrame.from_records(rows, columns=columns)

            cursor.close()
            self.connection.close()

            return df

        except Exception as e:
            print("خطا در اتصال یا اجرای کوئری:", e)

    def insert(self, insert_query, values):
        self.connection = pyodbc.connect(self.connection_string)
        cursor = self.connection.cursor()
        cursor.execute(insert_query, values)
        self.connection.commit()
        cursor.close()
        self.connection.close()

    def insert(self, insert_query):
        self.connection = pyodbc.connect(self.connection_string)
        cursor = self.connection.cursor()
        cursor.execute(insert_query)
        self.connection.commit()
        cursor.close()
        self.connection.close()
    
    def batch_insert(self, table_name : str,df : pd.DataFrame):
        self.connection = pyodbc.connect(self.connection_string)
        cursor = self.connection.cursor()
        columns = ",".join(df.columns)
        placeholders = ",".join("?" * len(df.columns))  
        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        data = [tuple(row) for row in df.to_numpy()]

        cursor.executemany(insert_query, data)
        self.connection.commit()
        cursor.close()
        self.connection.close()
        pass
