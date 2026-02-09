import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import fsolve
from datetime import datetime

# Read data
df = pd.read_csv('bonds_info.csv')

# Display data structure
print(df.head())
print(df.columns)

# Prepare trading date list
trading_dates = ['Jan5', 'Jan6', 'Jan7', 'Jan8', 'Jan9', 'Jan12', 'Jan13', 'Jan14', 'Jan15', 'Jan16']
price_columns = [f'Price_{date}' for date in trading_dates]

# Bond basic information
bond_names = df['Standard_Name'].tolist()
coupons = df['Coupon'].tolist()  # Annual coupon rate
years_to_maturity = df['Years_to_Maturity'].tolist()
face_value = 100  # Assume face value is 100

print(f"Total {len(bond_names)} bonds")
print(f"Maturities: {years_to_maturity}")

def calculate_ytm(price, coupon_rate, years_to_maturity, face_value=100, freq=2):
    """
    Calculate bond yield to maturity (YTM)
    Semi-annual coupon payment, semi-annual compounding
    """
    # Convert annual coupon to periodic coupon
    coupon_per_period = (coupon_rate * face_value) / freq
    periods = int(years_to_maturity * freq)  # Total number of payment periods
    
    def ytm_func(y):
        # y is periodic yield (semi-annual)
        pv = 0
        for t in range(1, periods + 1):
            pv += coupon_per_period / ((1 + y) ** t)
        pv += face_value / ((1 + y) ** periods)
        return pv - price
    
    # Initial guess: assume between 2-6%
    initial_guess = 0.02 / freq
    try:
        ytm_period = fsolve(ytm_func, initial_guess)[0]
        # Convert to annual YTM (bond equivalent yield)
        ytm_annual = ytm_period * freq
        return ytm_annual
    except:
        return np.nan

# Calculate YTM for each bond each day
ytm_data = pd.DataFrame(index=bond_names, columns=trading_dates)

for i, bond in enumerate(bond_names):
    coupon_rate = coupons[i]
    maturity = years_to_maturity[i]
    
    for date_idx, date in enumerate(trading_dates):
        price_col = f'Price_{date}'
        price = df.loc[i, price_col]
        
        if pd.notna(price):
            ytm = calculate_ytm(price, coupon_rate, maturity)
            ytm_data.loc[bond, date] = ytm
        else:
            ytm_data.loc[bond, date] = np.nan

# Convert to numeric type
ytm_data = ytm_data.astype(float)

print("YTM data sample:")
print(ytm_data.head())

def plot_yield_curves(ytm_data, years_to_maturity, trading_dates):
    """
    Plot yield curves for each day (overlay plot)
    """
    plt.figure(figsize=(12, 8))
    
    # Plot curve for each day
    for date_idx, date in enumerate(trading_dates):
        ytms = ytm_data[date].values
        
        # Use cubic spline interpolation
        if len(ytms[~pd.isna(ytms)]) >= 3:  # Need at least 3 points for interpolation
            valid_mask = ~pd.isna(ytms)
            valid_maturities = np.array(years_to_maturity)[valid_mask]
            valid_ytms = ytms[valid_mask]
            
            # Sort
            sort_idx = np.argsort(valid_maturities)
            valid_maturities = valid_maturities[sort_idx]
            valid_ytms = valid_ytms[sort_idx]
            
            # Create interpolation function
            if len(valid_maturities) >= 2:
                f = interpolate.interp1d(valid_maturities, valid_ytms, 
                                         kind='cubic', fill_value='extrapolate')
                
                # Generate smooth curve
                smooth_maturities = np.linspace(0.5, 5, 50)
                smooth_ytms = f(smooth_maturities)
                
                # Plot
                plt.plot(smooth_maturities, smooth_ytms * 100, 
                        label=f'{date}', linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Maturity (Years)', fontsize=12)
    plt.ylabel('Yield to Maturity (%)', fontsize=12)
    plt.title('Canadian Government Bond Yield Curves (January 2026)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig('yield_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# Call function to plot
plot_yield_curves(ytm_data, years_to_maturity, trading_dates)

def bootstrap_spot_rates(bond_data, date):
    """
    Calculate spot rate curve using bootstrap method
    bond_data: DataFrame containing all bond information
    date: trading date
    """
    # Sort by remaining maturity
    sorted_bonds = bond_data.copy()
    sorted_bonds = sorted_bonds.sort_values('Years_to_Maturity')
    
    # Initialize result dictionary
    spot_rates = {}
    
    # Process each bond
    for idx, row in sorted_bonds.iterrows():
        price_col = f'Price_{date}'
        price = row[price_col]
        coupon_rate = row['Coupon']
        maturity = row['Years_to_Maturity']
        
        if pd.isna(price):
            continue
        
        # Calculate periodic cash flows
        freq = 2  # Semi-annual payment
        periods = int(maturity * freq)
        coupon_per_period = (coupon_rate * 100) / freq
        
        # Calculate present value of known cash flows
        known_pv = 0
        for t in range(1, periods + 1):
            time_years = t / freq  # Convert to years
            
            if time_years in spot_rates:
                # Discount using known spot rate
                discount_rate = spot_rates[time_years]
                known_pv += coupon_per_period / ((1 + discount_rate) ** time_years)
            elif t == periods:
                # Last period (principal + final coupon)
                unknown_time = time_years
                unknown_cf = coupon_per_period + 100
            else:
                # Intermediate coupon with unknown spot rate (should not happen)
                continue
        
        # Solve for spot rate of last period
        def spot_rate_func(r):
            return known_pv + unknown_cf / ((1 + r) ** unknown_time) - price
        
        # Initial guess: assume near YTM
        initial_guess = 0.03
        try:
            spot_rate = fsolve(spot_rate_func, initial_guess)[0]
            spot_rates[unknown_time] = spot_rate
        except:
            # If solving fails, use approximation
            spot_rate = ((unknown_cf / (price - known_pv)) ** (1/unknown_time)) - 1
            spot_rates[unknown_time] = spot_rate
    
    return spot_rates

# Calculate spot rates for each day
all_spot_curves = {}

for date in trading_dates:
    spot_rates = bootstrap_spot_rates(df, date)
    all_spot_curves[date] = spot_rates
    print(f"{date} spot rates: {spot_rates}")

# Plot spot rate curves
def plot_spot_curves(all_spot_curves, trading_dates):
    plt.figure(figsize=(12, 8))
    
    for date in trading_dates:
        spot_rates = all_spot_curves[date]
        if spot_rates:
            maturities = sorted(spot_rates.keys())
            rates = [spot_rates[m] * 100 for m in maturities]  # Convert to percentage
            
            # Interpolation for smoothing
            if len(maturities) >= 3:
                f = interpolate.interp1d(maturities, rates, kind='cubic', fill_value='extrapolate')
                smooth_maturities = np.linspace(min(maturities), max(maturities), 50)
                smooth_rates = f(smooth_maturities)
                
                plt.plot(smooth_maturities, smooth_rates, label=date, linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Maturity (Years)', fontsize=12)
    plt.ylabel('Spot Rate (%)', fontsize=12)
    plt.title('Canadian Government Bond Spot Rate Curves (January 2026)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig('spot_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_spot_curves(all_spot_curves, trading_dates)

def calculate_forward_rates(spot_rates, date):
    """
    Calculate forward rates from spot rates
    Calculate 1-year forward rates for 1, 2, 3, 4 years ahead
    """
    forward_rates = {}
    
    # Get required spot rates
    # Assume we have 1, 2, 3, 4, 5 year spot rates
    # If not, interpolate from available spot rates
    
    # Create spot rate interpolation function
    if len(spot_rates) >= 2:
        maturities = sorted(spot_rates.keys())
        rates = [spot_rates[m] for m in maturities]
        spot_interp = interpolate.interp1d(maturities, rates, kind='cubic', fill_value='extrapolate')
        
        # Calculate forward rates (discrete compounding)
        # Formula: F_{t,T} = [(1+S_T)^T / (1+S_t)^t]^{1/(T-t)} - 1
        
        # 1-year forward rates
        for T in [2, 3, 4, 5]:  # Forward end time (years)
            t = 1  # Forward start time (years)
            
            try:
                S_t = spot_interp(t)
                S_T = spot_interp(T)
                
                # Calculate forward rate
                forward_rate = ((1 + S_T) ** T / (1 + S_t) ** t) ** (1/(T - t)) - 1
                forward_rates[f'1yr-{T-t}yr'] = forward_rate
            except:
                forward_rates[f'1yr-{T-t}yr'] = np.nan
    
    return forward_rates

# Calculate forward rates for each day
all_forward_curves = {}

for date in trading_dates:
    spot_rates = all_spot_curves[date]
    forward_rates = calculate_forward_rates(spot_rates, date)
    all_forward_curves[date] = forward_rates
    print(f"{date} forward rates: {forward_rates}")

# Plot forward rate curves
def plot_forward_curves(all_forward_curves, trading_dates):
    plt.figure(figsize=(12, 8))
    
    forward_terms = ['1yr-1yr', '1yr-2yr', '1yr-3yr', '1yr-4yr']
    
    # Prepare data for each day
    for date in trading_dates:
        forward_rates = all_forward_curves[date]
        if forward_rates:
            rates = []
            for term in forward_terms:
                rate = forward_rates.get(term, np.nan)
                if not pd.isna(rate):
                    rates.append(rate * 100)  # Convert to percentage
                else:
                    rates.append(np.nan)
            
            # Plot
            plt.plot(range(1, 5), rates, 'o-', label=date, linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Forward Term (Years)', fontsize=12)
    plt.ylabel('Forward Rate (%)', fontsize=12)
    plt.title('Canadian Government Bond Forward Rate Curves (1-Year Forward)', fontsize=14)
    plt.xticks([1, 2, 3, 4], ['1yr-1yr', '1yr-2yr', '1yr-3yr', '1yr-4yr'])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig('forward_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_forward_curves(all_forward_curves, trading_dates)

# Extract key maturity yields (1, 2, 3, 4, 5 years)
def extract_key_rates(all_spot_curves, trading_dates):
    """
    Extract key maturity rates from spot rate curves
    """
    key_maturities = [1.0, 2.0, 3.0, 4.0, 5.0]
    key_rates = {f'{m}yr': [] for m in key_maturities}
    
    for date in trading_dates:
        spot_rates = all_spot_curves[date]
        
        # Interpolation function
        if len(spot_rates) >= 2:
            maturities = sorted(spot_rates.keys())
            rates = [spot_rates[m] for m in maturities]
            spot_interp = interpolate.interp1d(maturities, rates, kind='cubic', fill_value='extrapolate')
            
            for m in key_maturities:
                try:
                    rate = spot_interp(m)
                    key_rates[f'{m}yr'].append(rate)
                except:
                    key_rates[f'{m}yr'].append(np.nan)
    
    return pd.DataFrame(key_rates, index=trading_dates)

# Extract key maturity spot rates
key_spot_rates = extract_key_rates(all_spot_curves, trading_dates)
print("Key maturity spot rates:")
print(key_spot_rates)

# Calculate log returns
def calculate_log_returns(rate_series):
    """
    Calculate log return series
    """
    log_returns = []
    for i in range(len(rate_series) - 1):
        if pd.notna(rate_series[i]) and pd.notna(rate_series[i+1]) and rate_series[i] != 0:
            log_return = np.log(rate_series[i+1] / rate_series[i])
            log_returns.append(log_return)
        else:
            log_returns.append(np.nan)
    return log_returns

# Calculate log returns for each maturity
yield_log_returns = pd.DataFrame()

for column in key_spot_rates.columns:
    series = key_spot_rates[column]
    log_returns = calculate_log_returns(series)
    yield_log_returns[column] = log_returns

print("Yield log returns:")
print(yield_log_returns)

# Calculate covariance matrix (remove NaN)
yield_cov_matrix = yield_log_returns.cov()
print("Yield log return covariance matrix:")
print(yield_cov_matrix)

# Extract forward rates
forward_rates_df = pd.DataFrame(all_forward_curves).T
print("Forward rates:")
print(forward_rates_df)

# Calculate forward rate log returns
forward_log_returns = pd.DataFrame()

for column in forward_rates_df.columns:
    series = forward_rates_df[column]
    log_returns = calculate_log_returns(series)
    forward_log_returns[column] = log_returns

print("Forward rate log returns:")
print(forward_log_returns)

# Calculate forward rate covariance matrix
forward_cov_matrix = forward_log_returns.cov()
print("Forward rate log return covariance matrix:")
print(forward_cov_matrix)

# PCA analysis
def perform_pca(cov_matrix):
    """
    Perform PCA analysis on covariance matrix
    """
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort (descending order)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate variance explained ratio
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    return eigenvalues, eigenvectors, explained_variance_ratio, cumulative_variance_ratio

# PCA for yields
yield_eigenvalues, yield_eigenvectors, yield_explained, yield_cumulative = perform_pca(yield_cov_matrix)

print("\nYield PCA results:")
print(f"Eigenvalues: {yield_eigenvalues}")
print(f"Eigenvectors (by column):\n{yield_eigenvectors}")
print(f"Variance explained ratio: {yield_explained}")
print(f"Cumulative variance explained ratio: {yield_cumulative}")

# PCA for forward rates
forward_eigenvalues, forward_eigenvectors, forward_explained, forward_cumulative = perform_pca(forward_cov_matrix)

print("\nForward rate PCA results:")
print(f"Eigenvalues: {forward_eigenvalues}")
print(f"Eigenvectors (by column):\n{forward_eigenvectors}")
print(f"Variance explained ratio: {forward_explained}")
print(f"Cumulative variance explained ratio: {forward_cumulative}")

# Plot PCA results
def plot_pca_results(eigenvalues, explained_variance_ratio, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Eigenvalues bar chart
    axes[0].bar(range(1, len(eigenvalues) + 1), eigenvalues)
    axes[0].set_xlabel('Principal Component', fontsize=12)
    axes[0].set_ylabel('Eigenvalue', fontsize=12)
    axes[0].set_title(f'{title} - Eigenvalues', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Variance explained ratio
    axes[1].plot(range(1, len(explained_variance_ratio) + 1), 
                 explained_variance_ratio * 100, 'o-', label='Individual')
    axes[1].plot(range(1, len(explained_variance_ratio) + 1), 
                 np.cumsum(explained_variance_ratio) * 100, 's-', label='Cumulative')
    axes[1].set_xlabel('Principal Component', fontsize=12)
    axes[1].set_ylabel('Variance Explained (%)', fontsize=12)
    axes[1].set_title(f'{title} - Variance Explained', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_pca.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_pca_results(yield_eigenvalues, yield_explained, 'Yield')
plot_pca_results(forward_eigenvalues, forward_explained, 'Forward Rate')

# Results interpretation
print("\n=== Results Interpretation ===")
print("\n1. Yield PCA analysis:")
print(f"Largest eigenvalue: {yield_eigenvalues[0]:.6f}")
print(f"Corresponding eigenvector: {yield_eigenvectors[:, 0]}")
print("Interpretation: The eigenvector corresponding to the largest eigenvalue typically represents")
print("the 'level factor', i.e., parallel shift of the yield curve.")
print("All elements with same sign and similar magnitude indicate the curve mainly moves in parallel.")

print(f"\nFirst principal component explains: {yield_explained[0]*100:.2f}% of variance")

print("\n2. Forward rate PCA analysis:")
print(f"Largest eigenvalue: {forward_eigenvalues[0]:.6f}")
print(f"Corresponding eigenvector: {forward_eigenvectors[:, 0]}")
print("Interpretation: The largest eigenvector for forward rates also typically represents the level factor,")
print("indicating that the main movement pattern of forward rate curve is also parallel shift.")

print(f"\nFirst principal component explains: {forward_explained[0]*100:.2f}% of variance")

print("\n3. General conclusion:")
print("For Canadian government bond market, the main movement pattern for both yield curve")
print("and forward rate curve is parallel shift, typically driven by monetary policy changes")
print("or macroeconomic expectation changes.")

# Save all results to CSV files
key_spot_rates.to_csv('key_spot_rates.csv')
yield_log_returns.to_csv('yield_log_returns.csv')
yield_cov_matrix.to_csv('yield_cov_matrix.csv')
forward_log_returns.to_csv('forward_log_returns.csv')
forward_cov_matrix.to_csv('forward_cov_matrix.csv')

# Save eigenvalues and eigenvectors
np.savetxt('yield_eigenvalues.csv', yield_eigenvalues, delimiter=',')
np.savetxt('yield_eigenvectors.csv', yield_eigenvectors, delimiter=',')
np.savetxt('forward_eigenvalues.csv', forward_eigenvalues, delimiter=',')
np.savetxt('forward_eigenvectors.csv', forward_eigenvectors, delimiter=',')

print("\nAll results saved to CSV files.")

# Add summary at the end
print("\n" + "="*60)
print("PCA Numerical Results Summary")
print("="*60)

print("\n【Yield PCA Results】")
print(f"Eigenvalues: {yield_eigenvalues}")
print(f"Variance Explained: {yield_explained}")
print(f"Cumulative Variance Explained: {yield_cumulative}")

print(f"\nFirst eigenvalue: {yield_eigenvalues[0]}")
print(f"First principal component explains: {yield_explained[0]*100:.2f}% of variance")

print("\n【Forward Rate PCA Results】")
print(f"Eigenvalues: {forward_eigenvalues}")
print(f"Variance Explained: {forward_explained}")
print(f"Cumulative Variance Explained: {forward_cumulative}")

print(f"\nFirst eigenvalue: {forward_eigenvalues[0]}")
print(f"First principal component explains: {forward_explained[0]*100:.2f}% of variance")