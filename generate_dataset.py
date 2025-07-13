import pandas as pd
import numpy as np
import random

# Set seed
random.seed(2025)
np.random.seed(2025)

# Define categories and factors
building_types = ["Residential", "Industrial", "Commercial"]
locations = ["Rural", "Suburban", "Urban"]
foundation_types = ["Pile", "Slab", "Concrete"]
roof_types = ["Flat", "Pitched", "Dome"]

type_cost_factor = {"Residential": 1.0, "Industrial": 1.2, "Commercial": 1.4}
location_cost_factor = {"Rural": 1.0, "Suburban": 1.15, "Urban": 1.3}
foundation_cost_factor = {"Pile": 1.0, "Slab": 1.1, "Concrete": 1.2}
roof_cost_factor = {"Flat": 1.0, "Pitched": 1.1, "Dome": 1.25}

num_rows = 5000

# Generate base data
df = pd.DataFrame({
    "building_type": np.random.choice(building_types, num_rows),
    "area_sqm": np.random.randint(250, 15001, num_rows),
    "floors": np.random.randint(1, 21, num_rows),
    "location": np.random.choice(locations, num_rows),
    "foundation_type": np.random.choice(foundation_types, num_rows),
    "roof_type": np.random.choice(roof_types, num_rows),
    "has_basement": np.random.choice([0, 1], num_rows, p=[0.6, 0.4]),
    "has_parking": np.random.choice([0, 1], num_rows, p=[0.3, 0.7]),
    "labor_rate": np.random.choice(range(3000, 15001, 100), num_rows)
})

# Compute total_cost
def compute_total_cost(row):
    # More realistic base calculation
    base_cost_per_sqm = random.uniform(800, 1500)  # Cost per sq meter
    
    cost = (
        row["area_sqm"] * 
        base_cost_per_sqm * 
        row["floors"] * 
        type_cost_factor[row["building_type"]] *
        location_cost_factor[row["location"]] *
        foundation_cost_factor[row["foundation_type"]] *
        roof_cost_factor[row["roof_type"]]
    )
    
    # Add labor costs (more realistic calculation)
    labor_cost = row["area_sqm"] * row["floors"] * (row["labor_rate"] / 10)  # Divided by 10 for realism
    cost += labor_cost

    # Add fixed costs for basement and parking
    if row["has_basement"]:
        cost += row["area_sqm"] * 300  # $300 per sqm for basement
    if row["has_parking"]:
        cost += row["area_sqm"] * 150  # $150 per sqm for parking

    # Add some random variation
    cost *= random.uniform(0.85, 1.15)
    
    # Ensure realistic cost range
    return int(np.clip(cost, 100_000, 50_000_000))  # $100K to $50M range

df["total_cost"] = df.apply(compute_total_cost, axis=1)

# Save the dataset
df.to_csv("realistic_construction_dataset.csv", index=False)

print(f"Dataset generated successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset statistics:")
print(df.describe())
