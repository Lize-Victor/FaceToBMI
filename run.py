from app import create_app
import threading
import backend.backend as backend
app = create_app()

if __name__ == '__main__':
    # Start the backend service
    backend_thread = threading.Thread(target=backend.run_backend)
    backend_thread.start()
    # Start the Flask app
    app_thread = threading.Thread(target=app.run, kwargs={'debug': False})
    app_thread.start()