import pandas as pd
import pandas.io.sql as psql
import MySQLdb as mdb

# Connect to the MySQL instance
db_host = 'localhost'
db_user = ''
db_pass = ''
db_name = ''

con = mdb.connect(db_host, db_user, db_pass, db_name)

# Select all of the historical close price for each bar of data in H1_price
sql = """SELECT h1p.price_date, h1p.close_price
         FROM symbol as sym
         INNER JOIN H1_price as h1p
         ON h1p.symbol_id = sym.id
         WHERE sym.ticker = 'eurusd'
         ORDER BY h1p.price_date ASC;"""

# Create a pandas df from the SQL query
eurusd = pd.read_sql(sql, con=con)

# Output the most recent close prices  of the df
print(eurusd.tail())
