"""Gunicorn settings for production (read automatically by `gunicorn app.app:app`)."""

import os

# Hosts such as Render pass the port to listen on in $PORT.
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# One worker keeps memory low on the free 512 MB plan (the app holds the final model
# and the precomputed JSON); threads handle concurrent requests.
workers = 1
threads = 4

# Load the model and caches once, before serving.
preload_app = True

timeout = 60

accesslog = "-"
errorlog = "-"
