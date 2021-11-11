import requests
import ast
import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
import time
import calendar

N_MONTHS = 6 # 120
N_SLEEP = 2 # Avoid getting ban 
STOCK_ID = '2330'
END_DAYTIME = datetime.date(2021, 11, 1) # Most recently 1st
FILL_IN = True # Fill in non-trading day

init_date = END_DAYTIME - relativedelta(months = N_MONTHS)

ans = []
for i in range(N_MONTHS + 1):
    d = init_date + relativedelta(months = i)
    url = 'https://www.twse.com.tw/exchangeReport/STOCK_DAY?date=' + d.__str__().replace("-", "") +'&stockNo=' + STOCK_ID
    print(f"Crawling {url}")
    res = requests.get(url)
    data = res.text

    for i in ast.literal_eval(data)['data']:
        (year, mon, day) = i[0].split('/')
        date_output = str(int(year) + 1911) + '-' + mon + '-' + day
        ans.append(date_output + ',' + i[-3] + '\n')
    time.sleep(N_SLEEP)


if FILL_IN:
    # Fill in value
    fill_in_list = []
    incre_date = init_date
    j = 0 # indice in ans
    while j < len(ans):
        y, m, d = ans[j][:10].split('-')
        sto_day = datetime.date(int(y), int(m), int(d))

        if incre_date == sto_day:
            fill_in_list.append(ans[j])
            incre_date = incre_date + relativedelta(days = 1)
            j += 1
        elif incre_date < sto_day:
            if j == 0:
                fill_in_list.append(str(incre_date) + ',nan\n')
            else:
                fill_in_list.append(str(incre_date) + ',' + ans[j-1][11:])
            incre_date = incre_date + relativedelta(days = 1)
        else: # cur_day > sto_day
            j += 1
    ans = fill_in_list

with open(f"{STOCK_ID}_{init_date}_{END_DAYTIME}.csv", 'w') as f:
    f.write("".join(ans))
