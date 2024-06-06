
from fastapi import FastAPI
import os

# Create an instance of the FastAPI class
app = FastAPI()

# Define a route using a decorator
@app.get("/")
def read_root():
    return {"message": "Hello, World! " + str(os.listdir())}

# Define another route
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
