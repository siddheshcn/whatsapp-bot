import logging

from app import create_app

app = create_app()
print(f"VERIFY_TOKEN: {app.config['VERIFY_TOKEN']}")

if __name__ == "__main__":
    logging.info("Flask app started")
    app.run(host="0.0.0.0", port=8000)
