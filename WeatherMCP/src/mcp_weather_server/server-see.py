import httpx
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.routing import Mount, Route
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from mcp.server import Server
from collections import defaultdict, Counter
from datetime import datetime
import uvicorn

# Create an MCP server
mcp = FastMCP("Weather")
import json

@mcp.tool()
async def get_weather(city: str) -> str:
    """Get weather information for a city, return json format for AI

    Dependencies:
        httpx
    """
    async with httpx.AsyncClient() as client:
        # Get coordinates using the Geocoding API
        geo_response = await client.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
        )
        if geo_response.status_code != 200 or "results" not in geo_response.json():
            return {"error": f"Could not retrieve coordinates for {city}."}

        geo_data = geo_response.json()["results"][0]
        latitude = geo_data["latitude"]
        longitude = geo_data["longitude"]

        # Get weather data using the Forecast API
        weather_response = await client.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,weather_code&timezone=auto&past_days=7"
        )

        if weather_response.status_code != 200:
            return {"error": f"Could not retrieve weather information for {city}."}

        weather_data = weather_response.json()

        weather_descriptions = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle",
            53: "Moderate drizzle", 55: "Dense drizzle", 56: "Light freezing drizzle",
            57: "Dense freezing drizzle", 61: "Slight rain", 63: "Moderate rain",
            65: "Heavy rain", 66: "Light freezing rain", 67: "Heavy freezing rain",
            71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
            77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
            82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }

        # Group data by date
        daily_temps = defaultdict(list)
        daily_weather_codes = defaultdict(list)

        for time_str, temp, code in zip(
            weather_data["hourly"]["time"],
            weather_data["hourly"]["temperature_2m"],
            weather_data["hourly"]["weather_code"]
        ):
            dt = datetime.fromisoformat(time_str)
            date_str = dt.date().isoformat()
            daily_temps[date_str].append(temp)
            daily_weather_codes[date_str].append(code)

        # Create summaries
        summaries = []
        for date in sorted(daily_temps.keys()):
            avg_temp = round(sum(daily_temps[date]) / len(daily_temps[date]))
            most_common_code = Counter(daily_weather_codes[date]).most_common(1)[0][0]
            description = weather_descriptions.get(most_common_code, "Unknown")
            dt = datetime.fromisoformat(date)
            day_of_week = dt.strftime("%A")
            summaries.append({
                "date": date,
                "day_of_week": day_of_week,
                "city": city,
                "weather": description,
                "temperature_celsius": avg_temp
            })

        return json.dumps(summaries, indent=2)

def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can server the provied mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

def main():
    mcp_server = mcp._mcp_server
    import argparse

    parser = argparse.ArgumentParser(description='Run MCP SSE-based server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    args = parser.parse_args()

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
