import csv
import random
from faker import Faker
import datetime

fake = Faker()

NUM_COMPANIES = 1000
NUM_COUNTRIES = 200
NUM_PRODUCT_TYPES = 50
TARGET_FILE_SIZE = 2 * 1024 * 1024

# Generate unique lists
companies = [fake.company() for _ in range(NUM_COMPANIES)]
countries = [fake.country() for _ in range(NUM_COUNTRIES)]
product_types = [
    f"{fake.word()} {random.choice(['Carbonate', 'Hydroxide', 'Chloride', 'Metal', 'Concentrate', 'Ore'])}"
    for _ in range(NUM_PRODUCT_TYPES)
]

# Define column names
columns = [
    "Date", "Company", "Country", "Product Type", "Production Volume (tonnes)",
    "Price (USD/tonne)", "Market Share (%)", "Purity (%)", "Capacity Utilization (%)",
    "Investment (million USD)", "R&D Spending (million USD)", "Environmental Impact Score",
    "Energy Consumption (MWh/tonne)", "Water Usage (m³/tonne)", "CO2 Emissions (kg/tonne)",
    "Recovery Rate (%)", "Extraction Method", "Export Volume (tonnes)", "Import Volume (tonnes)",
    "Stock Level (tonnes)", "Employee Count", "Safety Incidents"
]

# Define extraction methods
extraction_methods = ["Brine", "Hard Rock", "Clay", "Geothermal", "Recycling"]

# Generate data
def generate_row():
    date = fake.date_between(start_date="-5y", end_date="today")
    company = random.choice(companies)
    country = random.choice(countries)
    product_type = random.choice(product_types)
    production_volume = round(random.uniform(100, 10000), 2)
    price = round(random.uniform(5000, 20000), 2)
    market_share = round(random.uniform(0.01, 15), 2)
    purity = round(random.uniform(90, 99.9), 1)
    capacity_utilization = round(random.uniform(60, 100), 1)
    investment = round(random.uniform(1, 500), 2)
    rd_spending = round(random.uniform(0.5, 50), 2)
    environmental_impact = round(random.uniform(1, 10), 1)
    energy_consumption = round(random.uniform(5, 50), 2)
    water_usage = round(random.uniform(50, 500), 2)
    co2_emissions = round(random.uniform(500, 5000), 2)
    recovery_rate = round(random.uniform(30, 95), 1)
    extraction_method = random.choice(extraction_methods)
    export_volume = round(random.uniform(0, production_volume * 0.8), 2)
    import_volume = round(random.uniform(0, production_volume * 0.5), 2)
    stock_level = round(random.uniform(production_volume * 0.1, production_volume * 0.5), 2)
    employee_count = random.randint(50, 5000)
    safety_incidents = random.randint(0, 10)

    return [
        date.strftime("%Y-%m-%d"), company, country, product_type,
        production_volume, price, market_share, purity, capacity_utilization,
        investment, rd_spending, environmental_impact, energy_consumption,
        water_usage, co2_emissions, recovery_rate, extraction_method,
        export_volume, import_volume, stock_level, employee_count, safety_incidents
    ]

# Write data to CSV file
filename = "ltmidata.csv"
with open(filename, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(columns)
    file_size = 0
    while file_size < TARGET_FILE_SIZE:
        row = generate_row()
        writer.writerow(row)
        file_size = csvfile.tell()

print(f"Dataset generated: {filename}")
print(f"File size: {file_size / (1024 * 1024):.2f} MB")