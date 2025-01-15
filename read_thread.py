import shelve

with shelve.open('threads_db') as db:
    for key, value in db.items():
        print(f"WhatsApp ID: {key}")
        print(f"Thread ID: {value}")
        print("-" * 20)
