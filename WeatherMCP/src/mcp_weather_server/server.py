import httpx
import logging
from typing import Annotated
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from datetime import datetime
from . import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-weather")

# Create an MCP server
mcp = FastMCP("Weather")

async def get_coordinates(client: httpx.AsyncClient, city: str) -> tuple[float, float]:
    """
    Fetch the latitude and longitude for a given city using the Open-Meteo Geocoding API.

    :param client: An instance of httpx.AsyncClient
    :param city: The name of the city to fetch coordinates for
    :return: A tuple (latitude, longitude)
    :raises ValueError: If the coordinates cannot be retrieved
    """
    geo_response = await client.get(
        f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    )
    if geo_response.status_code != 200 or "results" not in geo_response.json():
        raise ValueError(f"Error: Could not retrieve coordinates for {city}.")

    geo_data = geo_response.json()["results"][0]
    return geo_data["latitude"], geo_data["longitude"]


@mcp.tool()
async def get_current_weather(city: Annotated[str ,Field(description="The name of the city to fetch weather information for, PLEASE NOTE English name only, if the parameter city isn't English please translate to English before invoking this function.")]) -> Annotated[str, Field(description="A string describing the current weather condition and temperature in the specified city, or an error message if the request fails.")]:

    """Get current weather information for a specified city.
    It extracts the current hour's temperature and weather code, maps
    the weather code to a human-readable description, and returns a formatted summary.
    """

    async with httpx.AsyncClient() as client:
        # Get coordinates using the Geocoding API
        try:
            try:
                latitude, longitude = await get_coordinates(client, city)
            except ValueError as e:
                return str(e)
            # Get weather data using the Forecast API
            url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,weather_code&timezone=GMT&forecast_days=1"
            logger.info(f"api: {url}")
            weather_response = await client.get(url)
            if weather_response.status_code != 200:
                return f"Error: Could not retrieve weather information for {city}."

            weather_data = weather_response.json()
            curIndex = utils.get_closest_utc_index(weather_data["hourly"]["time"])
            temperature = weather_data["hourly"]["temperature_2m"][curIndex]
            weather_code = weather_data["hourly"]["weather_code"][curIndex]
            relative_humidity = weather_data["hourly"]["relative_humidity_2m"][curIndex]
            dew_point = weather_data["hourly"]["dew_point_2m"][curIndex]

            description = utils.weather_descriptions.get(weather_code, "Unknown weather code")

            return f"The weather in {city} is {description} with a temperature of {temperature}°C, Relative humidity at 2 meters: {relative_humidity} %, Dew point temperature at 2 meters: {dew_point}"
        except Exception as e:
            logger.exception(f"API invoke error: {str(e)}")
            return f"API invoke error: {str(e)}"

@mcp.tool()
async def get_weather_by_datetime_range(
    city: Annotated[str, Field(description="The name of the city to fetch weather information for, PLEASE NOTE English name only, if the parameter city isn't English please translate to English before invoking this function.")],
    start_date: Annotated[str, Field(description="Start date in format YYYY-MM-DD, please follow ISO 8601 format")],
    end_date: Annotated[str, Field(description="End date in format YYYY-MM-DD , please follow ISO 8601 format")]
) -> Annotated[str, Field(description="A summary of hourly weather data for the given city and date range.")]:
    """
    Get weather information for a specified city between start and end dates.
    """

    async with httpx.AsyncClient() as client:
        try:
            try:
                latitude, longitude = await get_coordinates(client, city)
            except ValueError as e:
                return str(e)

            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={latitude}&longitude={longitude}"
                f"&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,weather_code"
                f"&timezone=GMT&start_date={start_date}&end_date={end_date}"
            )
            logger.info(f"api: {url}")
            response = await client.get(url)

            if response.status_code != 200:
                return f"Error: Could not retrieve weather information for {city}."

            data = response.json()
            times = data["hourly"]["time"]
            temperatures = data["hourly"]["temperature_2m"]
            humidities = data["hourly"]["relative_humidity_2m"]
            dew_points = data["hourly"]["dew_point_2m"]
            weather_codes = data["hourly"]["weather_code"]

            weather_data = []
            for time, temp, rh, dew,weather_code in zip(times, temperatures, humidities, dew_points,weather_codes):
                weather_data.append({
                    "time": time,
                    "temperature_c": temp,
                    "humidity_percent": rh,
                    "dew_point_c": dew,
                    "weather_description" : utils.weather_descriptions.get(weather_code, "Unknown weather code")
                })

            data_result = {
                "city": city,
                "start_date": start_date,
                "end_date": end_date,
                "weather": weather_data
            }

            return utils.format_get_weather_bytime(data_result)

        except Exception as e:
            logger.exception(f"API invoke error: {str(e)}")
            return f"API invoke error: {str(e)}"

@mcp.tool()
async def get_current_datetime(timezone_name: Annotated[str, Field(description="IANA timezone name (e.g., 'America/New_York', 'Europe/London'). Use UTC timezone if no timezone provided by the user.")]) -> str :
    """Get current time in specified timezone"""
    timezone = utils.get_zoneinfo(timezone_name)
    current_time = datetime.now(timezone)

    return utils.TimeResult(
        timezone=timezone_name,
        datetime=current_time.isoformat(timespec="seconds"),
    )

def main():
    mcp.run()

if __name__ == "__main__":
    main()
