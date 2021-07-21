import datetime
import csv
import pandas as pd
import MySQLdb as mdb


# Stores the current time
now = datetime.datetime.utcnow()


# Enter the csv_file name
csv_file = 'EURUSD_GMT+2_US-DST_H1.csv'
price_table = 'H1_price'

# Enter MySQL database credentials
db_host = 'localhost'
db_user = ''
db_pass = ''
db_name = ''


# Enter the details of the data vendor
dv_name = 'TrueFx'
dv_website_url = 'www.truefx.com'        # Dukascopy url: www.dukascopy.com
dv_created_date = now
dv_last_updated_date = now


# Enter the details of the pair you are inserting
sym_ticker = 'eurusd'
sym_instrument = 'spotfx'  # Stays same
sym_name = 'eurusd'
sym_gmt_offset = '2'
sym_timezone = 'us'
sym_created_date = now
sym_last_updated_date = now


def insert_data_vendor():
    """ Inserts data vendor information defined in globals into MySQL db. """
    con = mdb.connect(host=db_host, user=db_user, passwd=db_pass, db=db_name)

    column_str = "name, website_url, created_date, last_updated_date"
    insert_str = ("%s, " * 4)[:-2]
    final_str = "INSERT INTO data_vendor (%s) VALUES (%s)" % (column_str, insert_str)

    dv_data = (dv_name, dv_website_url, dv_created_date, dv_last_updated_date)

    # Using MySQL connection, insert information about the data vendor
    with con:
        cur = con.cursor()

        try:
            cur.execute(final_str, dv_data)
            con.commit()
        except mdb.IntegrityError:
            logging.warn("Failed to insert into data_vendor")
        finally:
            cur.close()


def insert_symbol():
    """ Inserts symbol information defined in globals into MySQL db. """
    con = mdb.connect(host=db_host, user=db_user, passwd=db_pass, db=db_name)

    column_str = "ticker, instrument, name, gmt_offset, timezone, created_date, last_updated_date"
    insert_str = ("%s, " * 7)[:-2]
    final_str = "INSERT INTO symbol (%s) VALUES (%s)" % (column_str, insert_str)

    sym_data = (sym_ticker, sym_instrument, sym_name, sym_gmt_offset, sym_timezone,
                sym_created_date, sym_last_updated_date)

    # Using MySQL connection, insert information about the data vendor
    with con:
        cur = con.cursor()

        try:
            cur.execute(final_str, sym_data)
            con.commit()
        except mdb.IntegrityError:
            logging.warn("Failed to insert into symbol")
        finally:
            cur.close()


def obtain_dv_and_sym_id():
    """ Obtains the data_vendor_id and symbol_id for use in inserting price data correctly. """

    con = mdb.connect(host=db_host, user=db_user, passwd=db_pass, db=db_name)

    query_str1 = "SELECT id FROM data_vendor WHERE name = '%s'" % dv_name
    print(query_str1)
    query_str2 = "SELECT id FROM symbol WHERE name = '%s'" % sym_name
    print(query_str2)

    id_list = []

    with con:
        cur = con.cursor()

        try:
            cur.execute(query_str1)
            dv_id = cur.fetchone()
            id_list.append(dv_id[0])

            cur.execute(query_str2)
            sym_id = cur.fetchone()
            id_list.append(sym_id[0])

        except mdb.IntegrityError:
            logging.warn("Failed to retrieve dv/sym_id")
        finally:
            cur.close()

    # print(id_list)
    return id_list


def create_forex_price_list():
    """ Reads in a csv containing MetaQuotes formatted bar data
        and formats it to the schema of the "tf_price" database. """

    csv = csv_file
    df = pd.read_csv(csv)
    # print(df.head())

    # Rename columns
    df.rename(columns={df.columns[0]: 'Date', df.columns[1]: 'Time',
                       df.columns[2]: 'Bar Open', df.columns[3]: 'Bar High',
                       df.columns[4]: 'Bar Low', df.columns[5]: 'Bar Close',
                       df.columns[6]: 'Volume'}, inplace=True)

    # Merge Date and Time column into one Datetime column
    # to match db schema
    df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    del df['Time']

    # Place None values where this missing data
    df = df.where((pd.notnull(df)), None)

    # Change df into list of tuples
    price_tuple_list = [tuple(x) for x in df.to_numpy()]

    # print(df)

    return price_tuple_list


def insert_forex_prices(price_data, vendor_id, symbol_id, table):
    """ Inserts the list of tuples of price data and inserts into the MySQL database. """

    con = mdb.connect(host=db_host, user=db_user, passwd=db_pass, db=db_name)

    # Amend the prices list to include vendor_id and symbol_id
    price_data = [(vendor_id, symbol_id, d[0], now, now,
                   d[1], d[2], d[3], d[4], d[5]) for d in price_data]

    # Create insert strings
    column_str = """data_vendor_id, symbol_id, price_date, created_date, last_updated_date,
                    open_price, high_price, low_price, close_price, volume"""
    insert_str = ("%s, " * 10)[:-2]
    final_str = "INSERT INTO %s (%s) VALUES (%s)" % (table, column_str, insert_str)

    # Using MySQL connection, carry out insert into price table
    with con:
        cur = con.cursor()

        try:
            cur.executemany(final_str, price_data)
            con.commit()
        except mdb.IntegrityError:
            logging.warn("Failed to insert into price table")
        finally:
            cur.close()


if __name__ == '__main__':

    """ Uncomment these methods when you need to insert new data vendor or symbol information. """
    print("Inserting data vendor info..")
    insert_data_vendor()
    print("Successfully inserted data vendor info!")
    print("Inserting symbol info..")
    insert_symbol()
    print("Successfully inserted symbol info!")

    id_list = []
    print("Obtaining data_vendor and symbol id..")
    id_list = obtain_dv_and_sym_id()
    print("Successfully obtained data vendor and symbol id!")
    print("Creating price data list..")
    prices = create_forex_price_list()
    print("Successfully created price data list!")
    print("Inserting price data into price table for %s" % sym_name)
    insert_forex_prices(prices, id_list[0], id_list[1], price_table)
    print("Successfully inserted price data into %s" % price_table)
