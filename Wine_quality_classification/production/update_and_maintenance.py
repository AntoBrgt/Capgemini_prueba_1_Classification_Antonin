import os
import time

def update_application():
    """Pull any update from the git
    """
    os.system("git pull origin main")

def restart_application():
    """Restart the app
    """

    os.system("docker-compose restart app")

def perform_maintenance():
    """Here I've to write all the operation that we need
    """
    pass

if __name__ == "__main__":
    while True:
        try:
            update_application()
            restart_application()
            perform_maintenance()
            print("Update and maintenance completed successfully.")
        except Exception as e:
            print("An error occurred during update and maintenance:", e)
        # Wait 24h
        time.sleep(24 * 3600) 