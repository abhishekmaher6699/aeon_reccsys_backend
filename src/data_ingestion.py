import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

host = os.getenv("POSTGRES_HOST")
port = "5432" 
database = os.getenv("POSTGRES_DB")
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASS")

table_name = "articles"

def setup_connection():

    try:
        connection = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password
        )

        cursor = connection.cursor()

        return connection, cursor
    except Exception as e:
        print(e)


def fetch_data():

    try:
        connection, cursor = setup_connection()
        print("Successfully connected to the aeon database.")
        
        query = f"SELECT * FROM {table_name}"
        print("Fetching data....")
        cursor.execute(query)

        colnames = [desc[0] for desc in cursor.description]

        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=colnames)

        df = df.drop(columns=['id', 'inserted_at', 'date', 'tags'])

        df.to_csv("./artifacts/data.csv", index=False)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        if connection:
            cursor.close()
            connection.close()

if __name__ == "__main__":
    fetch_data()
