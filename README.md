## Running the API

### 1. Install Dependencies

```bash
pip3 install -r requirements.txt or pip install -r requirements.txt
```

### 2. Start the Server

```bash
uvicorn main:app --reload
```

The FastAPI app will be available at [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---


pgAdmin: [http://localhost:5050](http://localhost:5050)
  -- Login: admin@admin.com / admin
  -- Add Server:
    -- Host: db
    -- Username: postgres
    -- Password: postgres



## Example API Call

Send a POST request with JSON payload:

```bash
curl -X POST http://127.0.0.1:8000/agent \
    -H "Content-Type: application/json" \
    -d '{"instruction": "Log task: Write weekly report and notify my team."}'
```