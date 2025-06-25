import uvicorn
from apps.api import create_app

if __name__ == "__main__":
    uvicorn.run("apps.api:create_app", host="0.0.0.0", port=8000, factory=True, reload=False) 