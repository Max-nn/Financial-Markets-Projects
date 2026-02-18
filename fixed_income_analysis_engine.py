
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from scipy.optimize import fsolve, brentq
import matplotlib.pyplot as plt
from datetime import datetime
from fredapi import Fred

# --- CONFIGURATION ---
# Replace with your actual API key
FRED_API_KEY = '1349ae70733091a5c1da47474fab7c6c'
fred = Fred(api_key=FRED_API_KEY)

# Dictionary mapping currencies to maturity years and their FRED tickers
# We use a mix of Treasury Rates (USD) and Interbank/Govt Benchmarks (EUR, GBP, CHF)

### WARNING : Most Tickers do not work, currently trying to find a solution ###
rf_map = {
    'USD': {
        0.25: 'DGS3MO',
        1.0: 'DGS1',
        2.0: 'DGS2',
        5.0: 'DGS5',
        7.0: 'DGS7',      # Added 7Y for better precision
        10.0: 'DGS10',
        20.0: 'DGS20',
        30.0: 'DGS30'
    },
    'EUR': {
        0.25: 'IR3TIB01EZM156N',  # 3-Month Euro Interbank Rate
        1.0: 'IR3TIB01EZM156N',   # Proxy: Using 3M rate for 1Y if 1Y missing (Optional, or remove)
        2.0: 'INTGSBDM020N',      # 2-Year German Govt Bond (Yield)
        5.0: 'INTGSBDM050N',      # 5-Year German Govt Bond (Yield) - NEW
        10.0: 'IRLTLT01DEM156N'   # 10-Year German Govt Bond (Long Term)
    },
    'GBP': {
        0.25: 'IR3TIB01GBM156N',  # 3-Month UK Interbank Rate
        2.0: 'INTGSGBM020N',      # 2-Year UK Govt Bond (Yield) - NEW
        5.0: 'INTGSGBM050N',      # 5-Year UK Govt Bond (Yield) - NEW
        10.0: 'IRLTLT01GBM156N'   # 10-Year UK Govt Bond
        # Note: 20Y/30Y UK Gilts are not free on FRED
    },
    'CHF': {
        0.25: 'IR3TIB01CHM156N',  # 3-Month Swiss Interbank Rate
        2.0: 'INTGSCHM020N',      # 2-Year Swiss Govt Bond (Yield) - NEW
        5.0: 'INTGSCHM050N',      # 5-Year Swiss Govt Bond (Yield) - NEW
        10.0: 'IRLTLT01CHM156N'   # 10-Year Swiss Govt Bond
    }
}

def get_yc_by_currency():
    """
    Fetches risk-free rates from FRED and interpolates a yield curve
    from 1 to 30 years for each currency.
    """
    all_curves = {}
    print("--- Starting Yield Curve Retrieval ---")

    # 1. Loop through each currency
    for currency, tickers_dict in rf_map.items():
        print(f"\nProcessing: {currency}")

        valid_maturities = []
        valid_rates = []

        # 2. Loop through specific maturities for this currency
        for maturity, ticker in tickers_dict.items():
            try:
                # Fetch series from FRED
                data = fred.get_series(ticker)

                # Safety checks
                if data is None or data.empty:
                    # print(f"  - No data for {ticker}")
                    continue

                # Get the most recent value
                latest_rate = data.iloc[-1]

                # Skip if value is NaN (Not a Number)
                if np.isnan(latest_rate):
                    continue

                # Store valid data (Convert percent to decimal: 5.0 -> 0.05)
                valid_maturities.append(maturity)
                valid_rates.append(latest_rate / 100.0)

            except Exception as e:
                print(f"  ! Error fetching {ticker}: {e}")

        # 3. Interpolation Logic
        if len(valid_rates) > 1:
            # CRITICAL: Sort data by maturity before interpolation
            # np.interp requires the x-coordinates (maturities) to be increasing
            sorted_pairs = sorted(zip(valid_maturities, valid_rates))
            sorted_mats = [x[0] for x in sorted_pairs]
            sorted_rates = [x[1] for x in sorted_pairs]

            # Define target maturities: 1 to 30 years
            target_maturities = np.arange(1, 31)

            # Calculate curve using linear interpolation
            # Note: For years beyond the max available data (e.g., >10Y for CHF),
            # np.interp will use the last known value (flat extrapolation).
            yield_curve = np.interp(target_maturities, sorted_mats, sorted_rates)

            # Store result
            all_curves[currency] = yield_curve
            print(f"  -> Success: Generated curve using {len(valid_rates)} data points.")
        else:
            print(f"  -> Failure: Not enough data points for {currency}")

    # 4. RETURN STATEMENT (Must be aligned with the 'for' loop, not inside it)
    return all_curves

# --- EXECUTION & PLOTTING ---

# 1. Run the function
yield_curves = get_yc_by_currency()

# 2. Display Data
print("\n--- Final Results (Rates for Year 1, 10, 30) ---")
df_results = pd.DataFrame(yield_curves, index=range(1, 31))
print(df_results.loc[[1, 2, 5, 10]]) # Display specific rows

# 3. Plotting the Curves
for curr in yield_curves.keys():
    plt.plot(range(1, 31), yield_curves[curr], label=curr, linewidth=2)

plt.title("Government Yield Curves (1-30 Years)")
plt.xlabel("Maturity (Years)")
plt.ylabel("Yield (Decimal)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

for currency in rf_map.keys():
    spot_rates = {}
    S1 = df_results.loc[1,currency]
    spot_rates[currency].append(S1)
    print(spot_rates)



















