from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np

url = "https://www.akc.org/reg/dogreg_stats.cfm"
r=requests.get(url)
data= r.text
soup = BeautifulSoup(data)

table = soup.find_all('table')[0]
rows = table.find_all('tr')



def get_cols_from_table_header(table_header):
    row_header_cols = table_header.find_all('td')
    return [header_col.get_text() for header_col in row_header_cols]

def read_table_as_data_frame(table_rows):
    table_header = table_rows[0]
    table_columns = get_cols_from_table_header(table_header)

    table_rows = table_rows[1:]

    #table_data = np.empty((len(table_rows), len(table_columns)))

    table_indices = np.arange(len(table_rows))

    table_data_frame = pd.DataFrame(index=table_indices, columns=table_columns)

    for i, table_row in enumerate(table_rows):
        table_data_frame.iloc[i] = [table_col.get_text() for table_col in table_row.find_all('td')]

    return table_data_frame;


akc_data = read_table_as_data_frame(rows)

print(akc_data)
akc_data.to_csv("akc_popularity.csv", encoding='utf-8')