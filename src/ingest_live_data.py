import requests
import pandas as pd
import logging

API_URL = "https://opensky-network.org/api/states/all"

def fetch_live_flights():

    flights = pd.DataFrame()   # always initialize

    try:

        response = requests.get(API_URL, timeout=10)

        if response.status_code != 200:
            logging.error("API request failed")
            return flights

        data = response.json()

        if "states" not in data or data["states"] is None:
            logging.error("No flight data returned")
            return flights

        columns = [
            "icao24","callsign","origin_country",
            "time_position","last_contact",
            "longitude","latitude",
            "baro_altitude","on_ground",
            "velocity","heading",
            "vertical_rate"
        ]

        flights = pd.DataFrame(data["states"])

        flights = flights.iloc[:, :len(columns)]

        flights.columns = columns

        current_time = pd.Timestamp.now()

        flights["Airline"] = flights["callsign"].str[:3]
        flights["Origin"] = flights["origin_country"]
        flights["Dest"] = flights["origin_country"]

        flights["FlightDate"] = current_time
        flights["ScheduledDepTime"] = current_time.hour

        logging.info(f"Fetched {len(flights)} flights")

        return flights

    except Exception as e:

        logging.error("Error fetching flight data")
        logging.error(str(e))

        return flights