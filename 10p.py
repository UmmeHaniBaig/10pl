import requests
import json
import csv
from datetime import datetime
import os
import sys
import time

# Configuration
CONFIG_FILE = "config.json"
CITY = "Karachi"
OUTPUT_DIR = "weather_data"

def load_config():
    """Load API key from config.json"""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            api_key = config.get('weatherapi_key')  # Updated key name
            if not api_key:
                raise ValueError("API key not found in config.json")
            return api_key
    except FileNotFoundError:
        print(f"❌ Error: {CONFIG_FILE} not found. Please create it with your API key.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in {CONFIG_FILE}.")
        sys.exit(1)

def fetch_weather_data(api_key):
    """Fetch weather data from WeatherAPI"""
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={CITY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "location": data["location"]["name"],
            "country": data["location"]["country"],
            "temperature": data["current"]["temp_c"],  # Temperature in Celsius
            "feels_like": data["current"]["feelslike_c"],  # Feels like in Celsius
            "weather": data["current"]["condition"]["text"],
            "humidity": data["current"]["humidity"],
            "wind_speed": data["current"]["wind_kph"],  # Wind speed in km/h
            "wind_dir": data["current"]["wind_dir"],
            "pressure": data["current"]["pressure_mb"],  # Pressure in mb
            "uv_index": data["current"]["uv"]  # UV index
        }
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {str(e)}")
        return None

def save_data(data):
    """Save data to JSON and CSV files"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON
    json_file = f"{OUTPUT_DIR}/weather_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # CSV (appends to existing file)
    csv_file = f"{OUTPUT_DIR}/weather_data.csv"
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
    
    return json_file, csv_file

if __name__ == "__main__":
    api_key = load_config()
    
    while True:
        weather_data = fetch_weather_data(api_key)
        
        if weather_data:
            json_path, csv_path = save_data(weather_data)
            print(f"✅ Data saved:\n- JSON: {json_path}\n- CSV: {csv_path}")
        else:
            sys.exit(1)
        
        # Wait for 10 seconds before the next update
        time.sleep(10)
