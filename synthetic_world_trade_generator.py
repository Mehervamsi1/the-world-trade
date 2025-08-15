# synthetic_world_trade_generator.py
# -----------------------------------
# Generates a 1,000,000-row synthetic world trade dataset (1975–2024)
# along with README and Data Dictionary files.
# Author: ChatGPT (Synthetic Data)
# License: MIT

import math, random
from pathlib import Path
import numpy as np
import pandas as pd

random.seed(7)
np.random.seed(7)

# ---------------------------------
# Parameters
# ---------------------------------
years = np.arange(1975, 2025)             # 50 years
n_reporters, n_partners = 80, 50          # disjoint sets
products = np.array(["AGRI","ENER","META","CHEM","MACH"])  # 5 categories
n_rows = len(years) * n_reporters * n_partners * len(products)  # 1,000,000

# Country pools (synthetic ISO3-like)
base_iso3 = [
    "USA","CAN","MEX","BRA","ARG","CHL","PER","COL","VEN","ECU","BOL","PRY","URY","GUF","SUR","GUY",
    "GBR","IRL","FRA","DEU","ESP","PRT","ITA","NLD","BEL","LUX","CHE","AUT","SWE","NOR","DNK","FIN","ISL",
    "POL","CZE","SVK","HUN","ROU","BGR","GRC","TUR","UKR","RUS","BLR","EST","LVA","LTU","SRB","HRV","SVN",
    "CHN","JPN","KOR","IND","PAK","BGD","LKA","NPL","MMR","THA","VNM","KHM","LAO","MYS","SGP","IDN","PHL",
    "AUS","NZL","PNG","FJI","SLB","VUT","WSM","TON","KIR","NRU",
    "ZAF","EGY","NGA","DZA","MAR","TUN","KEN","ETH","TZA","UGA","GHA","CIV","SEN","CMR","ZMB","ZWE","AGO","MOZ","BWA","NAM"
]
while len(base_iso3) < 130:
    base_iso3.append(f"X{len(base_iso3)+1:03d}")

reporters = np.array(base_iso3[:n_reporters])
partners  = np.array(base_iso3[-n_partners:])  # disjoint

continents = np.array(["Americas","Europe","Asia","Oceania","Africa"])
cont_map_rep = {iso: continents[i % len(continents)] for i, iso in enumerate(reporters)}
cont_map_par = {iso: continents[i % len(continents)] for i, iso in enumerate(partners)}

# Pseudo lat/lon by continent
cont_latrange = {"Americas":(-55,60), "Europe":(35,70), "Asia":(-10,55), "Oceania":(-45,10), "Africa":(-35,35)}
cont_lonrange = {"Americas":(-120,-35),"Europe":(-10,45),"Asia":(50,140),"Oceania":(110,180),"Africa":(-20,50)}

def rand_latlon(continent, rng):
    lat = rng.uniform(*cont_latrange[continent])
    lon = rng.uniform(*cont_lonrange[continent])
    return lat, lon

rng = np.random.default_rng(7)
rep_latlon = {iso: rand_latlon(cont_map_rep[iso], rng) for iso in reporters}
par_latlon = {iso: rand_latlon(cont_map_par[iso], rng) for iso in partners}

# Haversine vectorized
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# Distance matrix
rep_coords = np.array([rep_latlon[iso] for iso in reporters])
par_coords = np.array([par_latlon[iso] for iso in partners])
lat1 = rep_coords[:,0][:,None]
lon1 = rep_coords[:,1][:,None]
lat2 = par_coords[:,0][None,:]
lon2 = par_coords[:,1][None,:]
dist_mat = haversine(lat1, lon1, lat2, lon2)

# Macro fundamentals
def macro_for(country_list):
    n = len(country_list)
    gdp0 = rng.uniform(20, 2000, size=n)
    pop0 = rng.uniform(1, 150,   size=n)
    infl = rng.uniform(1.5, 8.0, size=n)
    gdp_g = rng.uniform(1.5, 5.0, size=n)/100.0
    pop_g = rng.uniform(0.2, 2.0, size=n)/100.0
    Y = len(years)
    shocks_gdp = rng.normal(0, 0.01, size=(Y,n))
    shocks_pop = rng.normal(0, 0.002, size=(Y,n))
    shocks_cpi = rng.normal(0, 0.01, size=(Y,n))
    gdp = np.zeros((Y,n))
    pop = np.zeros((Y,n))
    cpi = np.zeros((Y,n))
    gdp[0] = gdp0*(1+gdp_g+shocks_gdp[0])
    pop[0] = pop0*(1+pop_g+shocks_pop[0])
    cpi[0] = 50.0*(1+infl/100.0+shocks_cpi[0])
    for t in range(1,Y):
        gdp[t] = gdp[t-1]*(1+gdp_g+shocks_gdp[t])
        pop[t] = pop[t-1]*(1+pop_g+shocks_pop[t])
        cpi[t] = cpi[t-1]*(1+infl/100.0+shocks_cpi[t])
    return gdp, pop, cpi

gdp_rep, pop_rep, cpi_rep = macro_for(reporters)
gdp_par, pop_par, cpi_par = macro_for(partners)

# FTAs
pair_mask = rng.random((n_reporters, n_partners)) < 0.12
start_years = 1990 + rng.integers(0, 25, size=(n_reporters, n_partners))

# Tariffs & scaling
prod_base = {p: rng.uniform(2,15) for p in products}
theta = 1.2; beta_fta = 0.25; beta_tar = -0.03
prod_scale = {"AGRI":0.6,"ENER":1.4,"META":0.9,"CHEM":1.1,"MACH":1.6}

# Build index
Y, R, P, K = len(years), n_reporters, n_partners, len(products)
year_col     = np.repeat(years, R*P*K)
rep_idx_col  = np.tile(np.repeat(np.arange(R), P*K), Y)
par_idx_col  = np.tile(np.repeat(np.arange(P), K), Y*R)
prod_idx_col = np.tile(np.arange(K), Y*R*P)

rep_iso_col = reporters[rep_idx_col]
par_iso_col = partners[par_idx_col]
prod_col    = products[prod_idx_col]

# Lookups
gdp_r = gdp_rep[year_col - years[0], rep_idx_col]
gdp_p = gdp_par[year_col - years[0], par_idx_col]
pop_r = pop_rep[year_col - years[0], rep_idx_col]
pop_p = pop_par[year_col - years[0], par_idx_col]
cpi_r = cpi_rep[year_col - years[0], rep_idx_col]
cpi_p = cpi_par[year_col - years[0], par_idx_col]
dist_col = dist_mat[rep_idx_col, par_idx_col]

fta_active = (year_col >= start_years[rep_idx_col, par_idx_col]) & (pair_mask[rep_idx_col, par_idx_col])

years_since_1975 = (year_col - 1975).astype(float)
base_tar = np.vectorize(lambda p: prod_base[p])(prod_col)
tariff = base_tar - years_since_1975 * rng.uniform(0.02, 0.08) - (2.0 * fta_active.astype(float))
tariff = np.clip(tariff, 0.5, None)

shock = rng.lognormal(mean=0.0, sigma=0.6, size=n_rows)
fta_mult = np.exp(beta_fta) ** fta_active.astype(float)
tar_mult = np.exp(beta_tar * tariff)
gravity = (gdp_r * gdp_p) / (np.power(dist_col, theta) + 1.0)
prod_scale_arr = np.vectorize(lambda p: prod_scale[p])(prod_col)
value = gravity * fta_mult * tar_mult * shock * prod_scale_arr

qty = value * rng.uniform(5, 50)
unit_price = (value * 1e6) / np.maximum(qty, 1.0)

# DataFrame
df = pd.DataFrame({
    "year": year_col.astype(np.int16),
    "reporter_iso3": rep_iso_col,
    "partner_iso3": par_iso_col,
    "product_code": prod_col,
    "distance_km": np.round(dist_col, 1),
    "fta_active": fta_active.astype(np.int8),
    "adval_tariff_pct": np.round(tariff, 2),
    "reporter_gdp_bln": np.round(gdp_r, 3),
    "partner_gdp_bln": np.round(gdp_p, 3),
    "reporter_pop_m": np.round(pop_r, 3),
    "partner_pop_m": np.round(pop_p, 3),
    "reporter_cpi": np.round(cpi_r, 2),
    "partner_cpi": np.round(cpi_p, 2),
    "export_value_usd_mln": np.round(value, 3),
    "quantity_tonnes": np.round(qty, 3),
    "unit_price_usd_per_tonne": np.round(unit_price, 6),
})
assert len(df) == n_rows

# Save outputs locally
out_dir = Path(".")
df.to_csv(out_dir / "world_trade_synth_fast.csv", index=False)

readme = """# Synthetic World Trade Dataset (Fast Build, 1,000,000 rows)

This dataset contains **1,000,000 fully synthetic rows** for 1975–2024...
[truncated for brevity in this snippet]
"""
(out_dir / "README_world_trade_synth_fast.md").write_text(readme)

dd = """# Data Dictionary (Fast Build)

- year (int16): 1975–2024
...
"""
(out_dir / "DATA_DICTIONARY_world_trade_synth_fast.md").write_text(dd)

print("Dataset and docs generated in current folder.")

df = pd.read_csv('world_trade_synth_fast.csv')

df.to_csv("world_trade_synth_fast.csv.gz", index=False, compression="gzip")

