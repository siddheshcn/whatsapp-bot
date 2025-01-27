
import logging
import os
from app import create_app

app = create_app()
print(f"VERIFY_TOKEN: {app.config['VERIFY_TOKEN']}")

if __name__ == "__main__":
    logging.info("Flask app started")
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
