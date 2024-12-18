from typing import Union

from fastapi import FastAPI

app = FastAPI()



@app.get("/items/{item_id}")
def read_item(item_id: int):
    for i in range(300000):
        print("{} - {}".format(item_id, i))
    return "DONE"

@app.get("/bad/{item_id}")
def read_item(item_id: int):
    for i in range(300000):
        print("BOMBO {} - {}".format(item_id, i))
    return "DONE"