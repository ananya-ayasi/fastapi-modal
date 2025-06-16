from fastapi import FastAPI

app = FastAPI()

students= {
    1: {"name":"John", "age": 21 , "grade": "A"},
    2: {"name":"Jane", "age": 22, "grade": "B"},
}

@app.get("/")
def index():
    return {"name":"First Data"}  #JSON Object with a single key-value pair

app.get("/student/{id}")
def get_student(id:int):
    """
    Get student by ID.

    Args:
        id (int): The ID of the student to retrieve.

    Returns:
        dict: The student with the given ID.
    """
    return students[id]

