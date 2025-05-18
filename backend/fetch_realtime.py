import pandas
import json
from app import get_realtime 

def fetch_and_cache():
    realtime = get_realtime()

    recent_data = pandas.read_csv(
        '../dengue/public/recent_data.csv',
        names=['date','Cases'],
        header=0
    )
    global mdata
    mdata = pandas.read_csv('../dengue/public/websiteSmooth.csv')
    if mdata.index.name == "date":
        mdata = mdata.reset_index()

    new_data = pandas.DataFrame({
        "date": [realtime[2]],
        "Cases": [recent_data['Cases'][0]],
        "Rainfall": [realtime[1][2]],
        "Temperature": [realtime[1][0]],
        "RH": [realtime[1][1]],
        "searches1": [realtime[0][0]],
        "searches2": [realtime[0][1]]
    })
    mdata_new = pandas.concat([mdata, new_data], ignore_index=True)
    try:
        mdata_new["date"] = mdata_new["date"].dt.date
    except AttributeError:
        mdata_new["date"] = pandas.to_datetime(mdata_new["date"]).dt.date
    mdata_new.to_csv('../dengue/public/websiteSmooth.csv', index=False)

    # 3. Cache only the new row
    cache = {
        "date": realtime[2],
        "Cases":     recent_data['Cases'][0],
        "Rainfall":  realtime[1][2],
        "Temperature": realtime[1][0],
        "RH":        realtime[1][1],
        "searches1": realtime[0][0],
        "searches2": realtime[0][1]
    }
    with open('../dengue/public/realtime_cache.json', 'w') as f:
        json.dump(cache, f)

    print("fetched & cached realtime data")

if __name__ == "__main__":
    fetch_and_cache()