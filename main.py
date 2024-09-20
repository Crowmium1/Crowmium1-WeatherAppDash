import os
import requests
import pandas as pd
from io import BytesIO
from dataclasses import dataclass
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from datetime import datetime, timezone
import plotly.io as pio
from flask import Flask
from flask_caching import Cache 
import urllib.parse
import base64
from itertools import chain
from flask import send_from_directory


server = Flask(__name__)
server.config['CACHE_TYPE'] = 'SimpleCache'
server.config['CACHE_DEFAULT_TIMEOUT'] = 300 
cache = Cache(server)
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True)

# API Key and container names
API_KEY = 

# Blob Storage Configuration
STORAGE_CONTAINER_NAME = 
BLOB_CONTAINER_KEY = 
BLOB_CONNECTION_STRING = 

# Cosmos DB
COSMOS_DB_ENDPOINT = 
COSMOS_DB_KEY = 
COSMOS_DB_CONNECTION_STRING = 
COSMOS_DB_NAME = 
COSMOS_CONTAINER_NAME = 

# city = "Dublin"
# country = "IE"
client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
database = client.create_database_if_not_exists(id=COSMOS_DB_NAME)
container_name = database.create_container_if_not_exists(id=COSMOS_CONTAINER_NAME,partition_key=PartitionKey(path="/partitionKey"), offer_throughput=400)
DOWNLOAD_DIRECTORY = os.path.join(os.getcwd(), 'downloads')
########################################################################################################################
# Helper Functions
########################################################################################################################
# Function to get cached weather data
def get_cached_weather(city):
    return cache.get(city) 

# Function to cache weather data
def cache_weather_data(city, weather_data):
    cache.set(city, weather_data)

def json_to_dataframe(json_data):
    """Convert JSON data to DataFrame."""
    # Extract list of forecasts from JSON
    forecasts = json_data.get('list', [])
    
    # Convert to DataFrame
    df = pd.json_normalize(forecasts)
    
    # Normalize the 'weather' column to a separate DataFrame
    weather_df = pd.json_normalize(df['weather'].explode())
    
    # Drop the original 'weather' column and add new expanded columns
    df = df.drop(columns=['weather'])
    df = pd.concat([df, weather_df], axis=1)
    
    return df

def list_json_keys(json_obj, level=0, parent_key=''):
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            print(f"Level {level} - Key: {full_key}")
            # Recursively call for nested dictionaries or lists
            list_json_keys(value, level + 1, full_key)
    elif isinstance(json_obj, list):
        for index, item in enumerate(json_obj):
            full_key = f"{parent_key}[{index}]" if parent_key else f"[{index}]"
            print(f"Level {level} - List Index: {full_key}")
            # Recursively call for nested structures
            list_json_keys(item, level + 1, full_key)
    else:
        # Base case: print value (optional)
        print(f"Level {level} - Final Key: {parent_key} -> {json_obj}")

def save_plot_as_png(city, figure):
    # Logic to save plot as PNG file
    filename = f'{city}_historical_plot.png'
    figure.write_image(filename)
    return filename

def save_data_as_csv(city, data):
    # Logic to save data as CSV file
    df = pd.DataFrame(data)
    filename = f'{city}_historical_data.csv'
    df.to_csv(filename, index=False)
    return filename

@app.server.route('/download/<filename>')
def download_file(filename):
    try:
        return send_from_directory(DOWNLOAD_DIRECTORY, filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found.", 404

@app.server.route('/download_csv/<filename>')
def download_csv(filename):
    try:
        return send_from_directory(DOWNLOAD_DIRECTORY, filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found.", 404

########################################################################################################################
# PNG Storage 
########################################################################################################################
class BlobManager:
    def __init__(self, connection_string = BLOB_CONNECTION_STRING, container_name = "input"):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)

    def test_blob_connection(self):
        try:
            blob_list = self.container_client.list_blobs()
            print(f"Blobs in container '{self.container_client.container_name}':")
            for blob in blob_list:
                print(f"- {blob.name}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def upload_plot(self, fig, filename):
        try:
            bytes_io = BytesIO()
            fig.write_image(bytes_io, format="png")
            bytes_io.seek(0)
            blob_client = self.container_client.get_blob_client(filename)
            blob_client.upload_blob(bytes_io, overwrite=True)
            print(f"Uploaded '{filename}' to blob storage.")
        except Exception as e:
            print(f"Error uploading plot: {e}")
    
    def download_plot(self, filename):
        try:
            blob_client = self.container_client.get_blob_client(filename)
            return blob_client.download_blob().readall()
        except Exception as e:
            print(f"Error downloading plot: {e}")

########################################################################################################################
# Dataclass
########################################################################################################################
@dataclass
class WeatherData:
    id: str = ""
    city: str = ""
    country: str = ""
    temperature: float = 0.0
    weather_condition: str = ""
    wind_speed: float = 0.0
    humidity: float = 0.0
    pressure: float = 0.0
    timestamp: str = ""

    def format_for_db(self):
        return {
            'id': self.id,
            'name': self.city,
            'country': self.country,
            'temperature': self.temperature,
            'weather_condition': self.weather_condition,
            'wind_speed': self.wind_speed,
            'humidity': self.humidity,
            'pressure': self.pressure,
            'timestamp': self.timestamp
        }
    
    # New function to format the data for plotting
    def format_for_plot(self):
        return {
            'date_time': self.timestamp,
            'temperature': self.temperature
        }

########################################################################################################################
# API
########################################################################################################################
class WeatherAPI:
    @staticmethod
    def fetch_weather(city, country, API_KEY):
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city},{country}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return {}

    @staticmethod
    def fetch_forecast(city, country, API_KEY):
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={city},{country}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return {}

    @staticmethod
    def parse_weather(weather_data, forecast_data) -> WeatherData:
        country = weather_data['sys']['country']
        city = weather_data['name']
        temp = weather_data['main']['temp']
        description = weather_data['weather'][0]['description']
        wind_speed = weather_data['wind']['speed']
        date_time = forecast_data['list'][0]['dt_txt']
        humidity = weather_data['main']['humidity']
        pressure = weather_data['main']['pressure']
        return WeatherData(
            id=str(weather_data['id']), 
            city=city,
            country=country,
            temperature=temp,
            weather_condition=description,
            wind_speed=wind_speed,
            humidity=humidity,
            pressure=pressure,
            timestamp=date_time
        )
    
    @staticmethod
    def parse_forecast(forecast_data, num_forecasts) -> list:
        forecasts = []

        # Limit the number of forecasts based on the selected value
        for forecast in forecast_data['list'][:num_forecasts]:
            temp = forecast['main']['temp']
            description = forecast['weather'][0]['description']
            wind_speed = forecast['wind']['speed']
            date_time = datetime.fromtimestamp(forecast['dt'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            humidity = forecast['main']['humidity']
            pressure = forecast['main']['pressure']
            forecasts.append({
                'date_time': date_time,
                'temp': temp,
                'description': description,
                'wind_speed': wind_speed,
                'humidity': humidity,
                'pressure': pressure
            })
        return forecasts

    def print_short_summary(city, country, API_KEY = "de0b4371b47ae72b0b829d6aa66555ef"):
        weather_data_json = WeatherAPI.fetch_weather(city, country, API_KEY)
        forecast_data_json = WeatherAPI.fetch_forecast(city, country, API_KEY)
        if weather_data_json and 'main' in weather_data_json:
            weather_data = WeatherData(
                country=weather_data_json['sys']['country'],
                temperature=weather_data_json['main']['temp'],
                weather_condition=weather_data_json['weather'][0]['description'],
                wind_speed=weather_data_json['wind']['speed'],
                timestamp=forecast_data_json['list'][0]['dt_txt'] if 'list' in forecast_data_json else "",
                humidity=weather_data_json['main']['humidity'],
                pressure=weather_data_json['main']['pressure']
            )
            print(f"Current Weather in {city}, {weather_data.country} : {weather_data.weather_condition}")
            print(f"Temperature: {weather_data.temperature} °C")
            print(f"Humidity: {weather_data.humidity} %")
            print(f"Pressure: {weather_data.pressure} mbar")
            print(f"Wind: {weather_data.wind_speed} m/s")
            print(f"Forecast date and time: {weather_data.timestamp}")
        else:
            print("Error: Unable to fetch weather data.")

    def print_long_summary(city, country, API_KEY="de0b4371b47ae72b0b829d6aa66555ef"):
        weather_data = WeatherAPI.fetch_weather(city, country, API_KEY)
        forecast_data = WeatherAPI.fetch_forecast(city, country, API_KEY)

        if weather_data and 'main' in weather_data: 
            country = weather_data['sys']['country']
            temperature = weather_data['main']['temp']
            humidity = weather_data['main']['humidity']
            pressure = weather_data['main']['pressure']
            wind = weather_data['wind']['speed']
            description = weather_data['weather'][0]['description']
            date_time = datetime.fromtimestamp(weather_data['dt'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            date, time = date_time.split(' ')
            print(f"Date: {date}")
            print(f"Time: {time}")
            print(f"Current Condition in {city}, {country}: {description}")
            print(f"Temperature: {temperature} °C")
            print(f"Humidity: {humidity} %")
            print(f"Wind Speed: {pressure} m/s")
            print(f"Pressure: {wind} mbar")
            
            if forecast_data and 'list' in forecast_data:
                temps = [item['main']['temp'] for item in forecast_data['list']]
                humidities = [item['main']['humidity'] for item in forecast_data['list']]
                pressures = [item['main']['pressure'] for item in forecast_data['list']]
                wind = [item['wind']['speed'] for item in forecast_data['list']]
                avg_temp = sum(temps) / len(temps) if temps else 0
                avg_humidity = sum(humidities) / len(humidities) if humidities else 0
                avg_pressure = sum(pressures) / len(pressures) if pressures else 0
                avg_wind = sum(wind) / len(wind) if wind else 0
                print("\nForecast Averages:")
                print(f"Average Temperature: {avg_temp:.2f} °C")
                print(f"Average Humidity: {avg_humidity:.2f} %")
                print(f"Average Pressure: {avg_pressure:.2f} mbar")
                print(f"Average Pressure: {wind:.2f} m/s")
                print("\nDetailed Forecast (next few timestamps):")
                for forecast in forecast_data['list'][:5]:  # Limit to 5 entries
                    timestamp = forecast['dt_txt']
                    temp = forecast['main']['temp']
                    description = forecast['weather'][0]['description']
                    print(f"{timestamp} - {description}, Temp: {temp} °C")
            else:
                print("Error: Unable to fetch forecast data.")
        else:
            print("Error: Unable to fetch weather data.")

    def print_forecast_averages(self, city, country, API_KEY="de0b4371b47ae72b0b829d6aa66555ef"):
        # Fetch forecast data
        forecast_data = WeatherAPI.fetch_forecast(city, country, API_KEY)

        if forecast_data and 'list' in forecast_data:
            # Extract forecast values for temperature, humidity, and pressure
            temps = [item['main']['temp'] for item in forecast_data['list']]
            humidities = [item['main']['humidity'] for item in forecast_data['list']]
            pressures = [item['main']['pressure'] for item in forecast_data['list']]

            # Calculate averages
            avg_temp = sum(temps) / len(temps) if temps else 0
            avg_humidity = sum(humidities) / len(humidities) if humidities else 0
            avg_pressure = sum(pressures) / len(pressures) if pressures else 0

            # Print forecast averages
            print("Forecast Averages:")
            print(f"Average Temperature: {avg_temp:.2f} °C")
            print(f"Average Humidity: {avg_humidity:.2f} %")
            print(f"Average Pressure: {avg_pressure:.2f} mbar")
        else:
            print("Error: Unable to fetch forecast data.")

    @staticmethod
    def json_to_dataframe(json_data):
        """Convert JSON data to DataFrame."""
        forecasts = json_data.get('list', [])
        df = pd.json_normalize(forecasts)
        weather_df = pd.json_normalize(df['weather'].explode())
        df = df.drop(columns=['weather'])
        df = pd.concat([df, weather_df], axis=1)
        return df

########################################################################################################################
# COSMOS Database
########################################################################################################################
class WeatherDatabase:
    def __init__(self, endpoint, key, database_name, container_name):
        self.client = CosmosClient(endpoint, key)
        self.database = self.client.create_database_if_not_exists(id=database_name)
        self.container = self.database.create_container_if_not_exists(
            id=container_name,
            partition_key=PartitionKey(path="/partitionKey"),
            offer_throughput=400
        )
        print(f"Container '{container_name}' initialized or already exists.")
    
    # Initialize or create container
    def initialize_container(self):
        try:
            self.container = self.database.create_container_if_not_exists(
                id=self.container,
                partition_key=PartitionKey(path="/partitionKey"),
                offer_throughput=400
            )
            print(f"Container '{self.container}' initialized or already exists.")
        except Exception as e:
            print(f"Error initializing container: {e}")

    # Upsert weather data
    def upsert_weather(self, weather_data):
        try:
            self.container.upsert_item(weather_data)
            print("Weather data stored successfully")
        except Exception as e:
            print(f"Error upserting weather data: {e}")

    # Upsert forecast data
    def upsert_forecast(self, parsed_weather, parsed_forecast):
        for forecast in parsed_forecast:
            forecast_item = {
                'id': parsed_weather.id,  # Keep the weather data ID
                'country': parsed_weather.country,
                'city': parsed_weather.city,
                'temperature': forecast['temp'],
                'weather_condition': forecast['description'],
                'wind_speed': forecast['wind_speed'],
                'timestamp': forecast['date_time'],
                'humidity': forecast['humidity'],
                'pressure': forecast['pressure'],
            }
            try:
                self.container.upsert_item(forecast_item)
                print(f"Upserted forecast data for {forecast['date_time']} successfully")
            except exceptions.CosmosHttpResponseError as e:
                print(f"Failed to upsert forecast data for {forecast['date_time']}: {e}")

    # Get forecast/historical data
    def get_forecast(self, city: str, start_date: str = None, end_date: str = None) -> list[dict]:
        query = "SELECT * FROM c WHERE c.city = @city"
        parameters = [{"name": "@city", "value": city}]
        if start_date and end_date:
            # Ensure timestamp includes time for accurate querying
            query += " AND c.timestamp >= @start_date AND c.timestamp <= @end_date"
            parameters.extend([
                {"name": "@start_date", "value": f"{start_date} 00:00:00"},
                {"name": "@end_date", "value": f"{end_date} 23:59:59"}
            ])
            print(f"Querying Cosmos DB for city: {city}, from {start_date} to {end_date}")
        try:
            items = self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            )
            return list(items)
        except Exception as e:
            print(f"Error querying historical data: {e}")
            return []

    # Query temperature range
    def query_temp_range(self, city: str, min_temp: float, max_temp: float) -> list[dict]:
        query = """
        SELECT * FROM c 
        WHERE c.city = @city AND c.temperature >= @min_temp AND c.temperature <= @max_temp
        """
        parameters = [
            {"name": "@city", "value": city},
            {"name": "@min_temp", "value": min_temp},
            {"name": "@max_temp", "value": max_temp}
        ]
        print(f"Querying Cosmos DB for {city} with temperature between {min_temp}°C and {max_temp}°C")
        try:
            items = self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            )
            return list(items)
        except Exception as e:
            print(f"Error querying temperature range: {e}")
            return []
        
    def query_variable(self, city: str, variable: str, start_date: str = None, end_date: str = None) -> list[dict]:
            valid_variables = ["temperature", "wind_speed", "humidity", "pressure"]
            if variable not in valid_variables:
                raise ValueError(f"Invalid variable selected: {variable}. Choose from {valid_variables}")

            query = f"SELECT c.timestamp, c.{variable} FROM c WHERE c.city = @city"
            parameters = [{"name": "@city", "value": city}]
            
            if start_date and end_date:
                query += " AND c.timestamp >= @start_date AND c.timestamp <= @end_date"
                parameters.extend([
                    {"name": "@start_date", "value": start_date},
                    {"name": "@end_date", "value": end_date}
                ])
            print(f"Querying {variable} data from Cosmos DB for {city}, between {start_date} and {end_date}")
            
            try:
                items = self.container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True)
                return list(items)
            except Exception as e:
                print(f"Error querying {variable} data: {e}")
                return []

########################################################################################################################
#################################################################################################################
# Dashboard
#################################################################################################################
db = WeatherDatabase(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY, database_name=COSMOS_DB_NAME, container_name=COSMOS_CONTAINER_NAME)
db.initialize_container()
blob_manager = BlobManager(BLOB_CONNECTION_STRING, STORAGE_CONTAINER_NAME)

class DashboardConfig:
    default_city: str = "Dublin"
    default_country: str = "IE"
    graph_types: list = [
        {'label': 'Bar Chart', 'value': 'bar'},
        {'label': 'Line Chart', 'value': 'line'},
        {'label': 'Scatter Plot', 'value': 'scatter'}
    ]
    default_graph_type: str = "bar"
    date_range_start: str = str(datetime.now(timezone.utc).date())
    date_range_end: str = str(datetime.now(timezone.utc).date())

dashboard_config = DashboardConfig()

app.layout = html.Div([
    html.H1("Weather Dashboard", style={'text-align': 'center', 'margin-bottom': '30px'}),  # Main Title

    # Current Weather Section
    html.Div([
        html.H3("Current Weather Data"),
        dcc.Input(
            id='current-city-input',
            type='text',
            placeholder='Enter city',
            value=dashboard_config.default_city,
            style={'margin-right': '10px'}
        ),
        dcc.Input(
            id='current-country-input',
            type='text',
            placeholder='Enter country',
            value=dashboard_config.default_country,
            style={'margin-right': '10px'}
        ),
        dcc.Dropdown(
            id='graph-type-dropdown',
            options=dashboard_config.graph_types,
            value=dashboard_config.default_graph_type,
            style={'width': '50%', 'margin-bottom': '20px'}
        ),
        html.Button('Generate Graph', id='generate-graph-btn', n_clicks=0, style={'margin-top': '10px'}),
        
        html.Label("Select number of forecast days:", style={'margin-top': '20px'}),
        dcc.Slider(
            id='forecast-days-slider',
            min=1,
            max=5,
            step=1,
            marks={i: f'{i} days' for i in range(1, 6)},
            value=3
        ),
        dcc.Graph(id='current-weather-graph', style={'margin-top': '20px'}),  # Graph for current weather
    ]),

    # Historical Weather Section
    html.Div([
        html.H3("Historical Weather Data"),
        dcc.Input(
            id='historical-city-input',
            type='text',
            placeholder='Enter city',
            value=dashboard_config.default_city,
            style={'margin-right': '10px'}),
        dcc.DatePickerRange(
            id='historical-date-picker',
            start_date=dashboard_config.date_range_start,
            end_date=dashboard_config.date_range_end,
            display_format='YYYY-MM-DD',
            style={'margin-bottom': '20px'}),
        # Download buttons for PNG and CSV
        html.Button('Generate Historical Graph', id='generate-historical-graph-btn', n_clicks=0, style={'margin-top': '10px'}),
        dcc.Graph(id='historical-weather-graph', style={'margin-top': '20px'}),

        # Buttons
        html.A('Download Plot',id='download-plot-btn',download="",href="",target="_blank",style={'display': 'none', 'margin-top': '10px'}),
        html.A('Download CSV',id='download-csv-btn',href='',target="_blank",style={'margin-top': '10px'}),

        # Error or status message for historical section
        html.Div(id='historical-weather-section-message', style={'color': 'red', 'margin-top': '10px'})
    ], style={'padding': '20px', 'border': '1px solid #ccc', 'margin': '10px'}),  # Historical Weather Section ends

    # Query Weather Variable Section
    html.Div([
        html.H3("Query Weather Variable"),
        dcc.Input(id='variable-city-input', type='text', placeholder='Enter city', value='Dublin'),
        dcc.Dropdown(
            id='weather-variable-dropdown',
            options=[
                {'label': 'Temperature', 'value': 'temperature'},
                {'label': 'Wind Speed', 'value': 'wind_speed'},
                {'label': 'Humidity', 'value': 'humidity'},
                {'label': 'Pressure', 'value': 'pressure'}
            ],
            value='temperature',  # Default value
            style={'width': '50%'}
        ),
        dcc.DatePickerRange(
            id='variable-date-picker',
            start_date=dashboard_config.date_range_start,
            end_date=dashboard_config.date_range_end,
            display_format='YYYY-MM-DD'
        ),
        html.Button('Query Data', id='query-variable-btn', n_clicks=0),
        dcc.Graph(id='variable-weather-graph')
    ], style={'padding': '20px', 'border': '1px solid #ccc', 'margin': '10px'}),  # Query Weather Variable Section ends

    # Query by Temperature Range Section
    html.Div([
        html.H3("Query by Temperature Range"),
        dcc.Input(
            id='temp-city-input',
            type='text',
            placeholder='Enter city',
            value=dashboard_config.default_city,
            style={'margin-right': '10px'}
        ),
        dcc.Input(
            id='min-temp-input',
            type='number',
            placeholder='Min temperature (°C)',
            style={'margin-right': '10px'}
        ),
        dcc.Input(
            id='max-temp-input',
            type='number',
            placeholder='Max temperature (°C)',
            style={'margin-right': '10px'}
        ),
        html.Button('Query Data', id='query-temp-range-btn', n_clicks=0, style={'margin-top': '10px'}),
        dcc.Graph(id='temp-range-graph', style={'margin-top': '20px'})
    ], style={'padding': '20px', 'border': '1px solid #ccc', 'margin': '10px'})
])

# Comes from the API
@app.callback(
    Output('current-weather-graph', 'figure'),
    Input('generate-graph-btn', 'n_clicks'),
    State('current-city-input', 'value'),
    State('current-country-input', 'value'),
    State('graph-type-dropdown', 'value'),
    State('forecast-days-slider', 'value')
)
def update_weather_graph(n_clicks, city, country, graph_type, forecast_days):
    if n_clicks == 0 or not city or not country:
        return {}

    cache_key = f"{city}_{forecast_days}"
    cached_weather_data = get_cached_weather(cache_key)

    if cached_weather_data:
        parsed_forecast = cached_weather_data
        print(f"Using cached weather data for {city}")
    else:
        weather_data = WeatherAPI.fetch_weather(city, country, API_KEY)
        forecast_data = WeatherAPI.fetch_forecast(city, country, API_KEY)

        # Validate the API response before parsing
        if not weather_data or not forecast_data or 'list' not in forecast_data:
            print(f"Error fetching weather or forecast data for {city}.")
            return {} 
        
        parsed_weather = WeatherAPI.parse_weather(weather_data, forecast_data)
        parsed_forecast = WeatherAPI.parse_forecast(forecast_data, num_forecasts = forecast_days * 8)
        
        cache_weather_data(city, parsed_forecast)
        print(f"Fetched and cached new forecast data for {city} for ({forecast_days} days)")

        parsed_weather = WeatherAPI.parse_weather(weather_data, forecast_data)
        parsed_forecast = WeatherAPI.parse_forecast(forecast_data, num_forecasts=forecast_days * 8)

        db.upsert_weather(parsed_weather.format_for_db())
        db.upsert_forecast(parsed_weather, parsed_forecast)

    # Use parsed forecast data for the graph
    temps = [forecast['temp'] for forecast in parsed_forecast]
    dates = [forecast['date_time'] for forecast in parsed_forecast]

    # Check that dates and temps are flat lists
    if not dates or not temps:
        print("No valid data for dates or temperatures.")
        return {}

    if graph_type == 'bar':
        fig = px.bar(x=dates, y=temps, title=f'Temperature in {city}', labels={'x': 'Date', 'y': 'Temperature (°C)'})
    else:
        fig = px.line(x=dates, y=temps, title=f'Temperature in {city}', labels={'x': 'Date', 'y': 'Temperature (°C)'})
    blob_manager = BlobManager(BLOB_CONNECTION_STRING, STORAGE_CONTAINER_NAME)
    blob_manager.upload_plot(fig, f'{city}_weather_{n_clicks}.png')

    return fig

@app.callback(
    [
        Output('historical-weather-graph', 'figure'),
        Output('historical-weather-section-message', 'children')
    ],
    Input('generate-historical-graph-btn', 'n_clicks'),
    State('historical-city-input', 'value'),
    State('historical-date-picker', 'start_date'),
    State('historical-date-picker', 'end_date')
)
def update_historical_graph(n_clicks, city, start_date, end_date):
    if n_clicks == 0 or not city or not start_date or not end_date:
        return {}, "Please generate a historical graph."

    # Query CosmosDB for historical data
    historical_data = db.get_forecast(city, start_date, end_date)
    print("Printing historical data:", historical_data)

    if not historical_data:
        print("No historical data available.")
        return {}, "No historical data found for the selected date range."

    # Convert historical data to DataFrame
    df = pd.DataFrame(historical_data)
    if 'timestamp' not in df.columns or 'temperature' not in df.columns:
        print("Required columns are missing.")
        return {}, "Required data fields are missing."

    df_filtered = df[['timestamp', 'temperature']]
    print("Filtered DataFrame:", df_filtered)

    # Check if the DataFrame is empty
    if df_filtered.empty:
        print("DataFrame is empty, no data to plot.")
        return {}, "No data available to plot."

    # Plot the data
    fig = px.line(df_filtered, x='timestamp', y='temperature',
                  title=f'Temperature in {city} from {start_date} to {end_date}',
                  labels={'timestamp': 'Timestamp', 'temperature': 'Temperature (°C)'})

    # Upload the graph to Blob Storage
    filename = f'{city}_historical_weather_{n_clicks}.png'
    blob_manager.upload_plot(fig, filename)
    print(f"Graph uploaded as {filename}")

    # Generate a downloadable link
    try:
        blob_data = blob_manager.download_plot(filename)
        if blob_data:
            # Use the Flask route to download the plot
            href = f"/download/{filename}"
            message = "Graph generated and uploaded successfully."
        else:
            href = ""
            message = "Graph uploaded, but failed to generate download link."
    except Exception as e:
        print(f"Error generating download link: {e}")
        href = ""
        message = "Graph uploaded, but failed to generate download link."

    return fig, message

@app.callback(
    Output('temp-range-graph', 'figure'),
    Input('query-temp-range-btn', 'n_clicks'),
    State('temp-city-input', 'value'),
    State('min-temp-input', 'value'),
    State('max-temp-input', 'value')
)
def query_temperature_range(n_clicks, city, min_temp, max_temp):
    if n_clicks == 0 or not city or min_temp is None or max_temp is None:
        return {}
    
    cache_key = f"{city}_temp_{min_temp}_{max_temp}"
    cached_data = get_cached_weather(cache_key)
    
    if cached_data:
        queried_data = cached_data
        print(f"Using cached temperature range data for {city}: {min_temp}°C to {max_temp}°C")
    else:
        # Query CosmosDB for weather data in the temperature range
        queried_data = db.query_temp_range(city, min_temp, max_temp)
        if not queried_data:
            return {}

        cache_weather_data(cache_key, queried_data)
        print(f"Fetched and cached temperature range data for {city}: {min_temp}°C to {max_temp}°C")

    dates = [item['timestamp'] for item in queried_data]
    temps = [item['temperature'] for item in queried_data]

    fig = px.line(x=dates, y=temps, title=f'Temperature in {city} between {min_temp}°C and {max_temp}°C',
                  labels={'x': 'Timestamp', 'y': 'Temperature (°C)'})

    blob_manager = BlobManager(BLOB_CONNECTION_STRING, STORAGE_CONTAINER_NAME)
    blob_manager.upload_plot(fig, f'{city}_temp_range_{min_temp}_{max_temp}_{n_clicks}.png')
    return fig

# @app.callback(
#     Output('temp-range-graph', 'figure'),
#     Input('query-temp-range-btn', 'n_clicks'),
#     State('temp-city-input', 'value'),
#     State('min-temp-input', 'value'),
#     State('max-temp-input', 'value')
# )
# def update_temp_range_graph(n_clicks, city, min_temp, max_temp):
    if n_clicks == 0 or not city or min_temp is None or max_temp is None:
        return {}

    # Query the database for temperature range data
    temperature_data = db.query_temp_range(city, min_temp, max_temp)

    if not temperature_data:
        return {
            'data': [],
            'layout': {
                'title': 'No Data Available',
                'xaxis': {'title': 'Timestamp'},
                'yaxis': {'title': 'Temperature (°C)'}
            }
        }

    # Convert the result to a DataFrame for plotting
    df = pd.DataFrame(temperature_data)
    df_filtered = df[['timestamp', 'temperature']]

    if df_filtered.empty:
        return {
            'data': [],
            'layout': {
                'title': 'No Data Available',
                'xaxis': {'title': 'Timestamp'},
                'yaxis': {'title': 'Temperature (°C)'}
            }
        }

    # Plot the data
    fig = px.line(df_filtered, x='timestamp', y='temperature',
                  title=f'Temperature in {city} between {min_temp}°C and {max_temp}°C',
                  labels={'timestamp': 'Timestamp', 'temperature': 'Temperature (°C)'})

    return fig

# @app.callback(
#     Output('variable-weather-graph', 'figure'),
#     Input('query-variable-btn', 'n_clicks'),
#     State('variable-city-input', 'value'),
#     State('weather-variable-dropdown', 'value'),
#     State('variable-date-picker', 'start_date'),
#     State('variable-date-picker', 'end_date')
# )
# def update_variable_graph(n_clicks, city, variable, start_date, end_date):
#     if n_clicks == 0 or not city or not variable or not start_date or not end_date:
#         return {}

#     # Query the data from the Cosmos DB
#     queried_data = db.query_variable(city, variable, start_date, end_date)

#     if not queried_data:
#         return {}  # Return empty graph if no data is found

#     # Extract timestamps and the selected variable values
#     dates = [item['timestamp'] for item in queried_data]
#     values = [item[variable] for item in queried_data]

#     # Create the line chart
#     fig = px.line(x=dates, y=values, title=f'{variable.capitalize()} in {city} from {start_date} to {end_date}',
#                   labels={'x': 'Timestamp', 'y': variable.capitalize()})
#     blob_manager = BlobManager(BLOB_CONNECTION_STRING, STORAGE_CONTAINER_NAME)
#     blob_manager.upload_plot(fig, f'{city}_{variable}_graph_{n_clicks}.png')
#     return fig

@app.callback(
    [Output('download-plot-btn', 'href'),
     Output('download-csv-btn', 'href')],
    [Input('generate-historical-graph-btn', 'n_clicks'),
     Input('download-plot-btn', 'n_clicks'),
     Input('download-csv-btn', 'n_clicks')],
    [State('historical-city-input', 'value'),
     State('historical-weather-graph', 'figure')]
)
def handle_downloads(generate_clicks, plot_clicks, csv_clicks, city, figure):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    plot_url = dash.no_update
    csv_url = dash.no_update

    if trigger_id == 'generate-historical-graph-btn' and generate_clicks > 0:
        if city and generate_clicks:
            plot_url = f"/download/{city}_historical_weather_{generate_clicks}.png"

    elif trigger_id == 'download-plot-btn' and plot_clicks > 0:
        if city and generate_clicks:
            plot_url = f"/download/{city}_historical_weather_{generate_clicks}.png"

    elif trigger_id == 'download-csv-btn' and csv_clicks > 0:
        historical_data = db.get_forecast(city, dashboard_config.date_range_start, dashboard_config.date_range_end)
        if historical_data:
            df = pd.DataFrame(historical_data)
            filename = f"{city}_historical_data_{generate_clicks}.csv"
            csv_path = os.path.join(DOWNLOAD_DIRECTORY, filename)
            df.to_csv(csv_path, index=False, encoding='utf-8')
            csv_url = f"/download_csv/{filename}"
        else:
            csv_url = ""

    return plot_url, csv_url


#################################################################################################################
# Scheduling and Monitoring
#################################################################################################################
# # Define the job to run every 10 minutes
# def job():
#     cities = ['London', 'New York', 'Tokyo']
    
#     for city in cities:
#         data = WeatherAPI.fetch_weather(city, "US")
#         weather_data = WeatherAPI.parse_weather_data(data)
#         if weather_data:
#             WeatherDatabase.store_weather_data(WeatherData.format_for_db(weather_data))
#             print(f"Weather data for {city} updated")

# # Schedule the job
# schedule.every(10).minutes.do(job)

# # Run the scheduled job continuously
# def run_schedule():
#     while True:
#         schedule.run_pending()
#         time.sleep(1)

# # Run the scheduler in a separate thread
# scheduler_thread = threading.Thread(target=run_schedule, daemon=True)
# scheduler_thread.start()

########################################################################################################################
# Run
########################################################################################################################
# Run app
if __name__ == '__main__':
    app.run_server(debug=True)