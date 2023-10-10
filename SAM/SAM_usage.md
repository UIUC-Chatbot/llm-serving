It runs on port 5003

```bash
conda activate torch_116

gunicorn -w 2 -t 300 -b 127.0.0.1:5003 production_flask_SAM:app
```