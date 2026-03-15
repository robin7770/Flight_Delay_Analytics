import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingest_live_data import fetch_live_flights
from src.preprocess import FlightPreprocessor
from src.anomaly_detection import FlightAnomalyDetector


# ------------------------------------------------
# PAGE SETTINGS
# ------------------------------------------------

st.set_page_config(page_title="Flight Analytics Dashboard", layout="wide")

st.title("✈ Real-Time Flight Analytics Dashboard")

st.write(
"This dashboard shows live aircraft data and simple insights about global flights."
)

# ------------------------------------------------
# FETCH DATA
# ------------------------------------------------

pre = FlightPreprocessor()

st.write("Fetching live flight data...")

flights = fetch_live_flights()

if flights.empty:

    st.warning("No flight data available right now.")

else:

    data = pre.preprocess(flights)

    # ------------------------------------------------
    # BASIC CLEANING
    # ------------------------------------------------

    data = data.dropna(subset=["latitude", "longitude"])

    data["callsign"] = data["callsign"].fillna("Unknown")

    # Extract airline code from callsign
    data["AirlineCode"] = data["callsign"].astype(str).str[:3]

    detector = FlightAnomalyDetector()

    data = detector.detect(data)

    anomalies = data[data["anomaly"] == -1]

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Flights", len(data))
    col2.metric("Airlines", data["AirlineCode"].nunique())
    col3.metric("Anomalous Flights", len(anomalies))

    import plotly.graph_objects as go

    fig = go.Figure()

    # Normal flights
    normal = data[data["anomaly"] == 1]

    fig.add_trace(
        go.Scattermapbox(
            lat=normal["latitude"],
            lon=normal["longitude"],
            mode="markers",
            marker=dict(size=7, color="blue"),
            text=normal["callsign"],
            name="Normal Flights"
        )
    )

    # Anomalous flights
    abnormal = data[data["anomaly"] == -1]

    fig.add_trace(
        go.Scattermapbox(
            lat=abnormal["latitude"],
            lon=abnormal["longitude"],
            mode="markers",
            marker=dict(size=10, color="red"),
            text=abnormal["callsign"],
            name="Anomalous Flights"
        )
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=1,
        mapbox_center={"lat":20, "lon":0},
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Flight Anomalies")

    st.dataframe(
        anomalies[[
            "callsign",
            "velocity",
            "baro_altitude",
            "vertical_rate"
        ]]
    )

    # ------------------------------------------------
    # OVERVIEW METRICS
    # ------------------------------------------------

    st.subheader("Live Flight Overview")

    col1, col2 = st.columns(2)

    col1.metric("Total Flights Detected", len(data))
    col2.metric("Unique Airline Codes", data["AirlineCode"].nunique())

    # ------------------------------------------------
    # CHART 1 : TOP AIRLINES
    # ------------------------------------------------

    st.subheader("Top Airlines Currently Flying")

    airline_counts = (
        data["AirlineCode"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    airline_counts.columns = ["Airline Code", "Number of Flights"]

    fig_airlines = px.bar(
        airline_counts,
        x="Airline Code",
        y="Number of Flights",
        title="Top 10 Airlines by Number of Flights",
        text="Number of Flights",
        color="Number of Flights",
        color_continuous_scale=["#4CAF50", "#F44336"]
    )

    fig_airlines.update_layout(
        xaxis_title="Airline Code",
        yaxis_title="Number of Flights"
    )

    st.plotly_chart(fig_airlines, use_container_width=True)

    st.info(
        "Each bar shows how many aircraft from each airline are currently flying. "
        "Airline codes are derived from the first 3 letters of the flight callsign."
    )

    # ------------------------------------------------
    # CHART 2 : ALTITUDE VS VELOCITY
    # ------------------------------------------------

    st.subheader("Aircraft Speed vs Altitude")

    fig_alt = px.scatter(
        data,
        x="baro_altitude",
        y="velocity",
        color="on_ground",
        hover_data=["callsign", "AirlineCode"],
        title="Relationship Between Aircraft Altitude and Speed"
    )

    fig_alt.update_layout(
        xaxis_title="Altitude (meters)",
        yaxis_title="Velocity (m/s)"
    )

    st.plotly_chart(fig_alt, use_container_width=True)

    st.info(
        "Each point represents an aircraft. Blue points indicate flights in the air, "
        "while red points represent aircraft on the ground."
    )

    # ------------------------------------------------
    # CHART 3 : LIVE FLIGHT MAP
    # ------------------------------------------------

    st.subheader("Live Aircraft Locations Around the World")

    fig_map = px.scatter_mapbox(
        data,
        lat="latitude",
        lon="longitude",
        hover_name="callsign",
        hover_data=["AirlineCode"],
        zoom=1,
        height=600
    )

    fig_map.update_layout(
        mapbox_style="open-street-map",
        title="Global Aircraft Positions"
    )

    st.plotly_chart(fig_map, use_container_width=True)

    st.info(
        "Hover over a point on the map to see the flight number and airline code."
    )