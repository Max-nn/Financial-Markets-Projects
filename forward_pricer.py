# On commence par importer les packages requis pour le projet
import yfinance as yf
import pandas as pd
import numpy as np
import math
from fredapi import Fred

# REMPLACEZ CECI PAR VOTRE CLÉ API FRED (Gratuite)
FRED_API_KEY = '1349ae70733091a5c1da47474fab7c6c'

# --- ÉTAPE 1 : ACQUISITION DES DONNÉES ---

def get_market_data(ticker_symbol):
    """
    Récupère le prix spot, le type d'actif et calcule le rendement du dividende.
    """
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        # Utilisation de fast_info pour une exécution rapide
        info = ticker_obj.info
        fast_info = ticker_obj.fast_info

        S0 = fast_info['lastPrice']
        asset_type = info.get('quoteType')
        currency = info.get('currency', 'USD')

        # Calcul du Dividend Yield continu (q)
        if asset_type == 'EQUITY':
            divs = ticker_obj.get_dividends()
            if not divs.empty:
                # Correction du décalage horaire pour la comparaison Pandas
                divs_naive = divs.copy()
                divs_naive.index = divs_naive.index.tz_localize(None)

                # Filtrage sur la dernière année glissante
                last_year = divs_naive[divs_naive.index > (pd.Timestamp.now() - pd.DateOffset(years=1))]
                # Passage en taux continu : ln(1 + div/price)
                div_yield = math.log(1 + (last_year.sum() / S0))
            else:
                div_yield = 0.0
        else:
            # Pour les ETF/Indices, on récupère la donnée brute (souvent en décimal)
            div_yield = info.get('dividendYield', 0)/100 or 0.0

        return S0, div_yield, ticker_obj, currency, asset_type
    except Exception as e:
        print(f" Erreur de sourcing pour {ticker_symbol} : {e}")
        return None, None, None, None, None


def get_rf_rate_by_currency(currency_code, maturity_days):
    """
    Récupère le taux sans risque par interpolation (Source: Yahoo Finance pour USD, FRED pour autres).
    """
    fred = Fred(api_key=FRED_API_KEY)
    target_years = maturity_days / 365.0

    # Dictionnaire des tickers
    rf_map = {
        'USD': {
            0.25: '^IRX',
            5.0: '^FVX',
            10.0: '^TNX'
        },
        'EUR': {
            2.0: 'INTGSBDM020N',
            5.0: 'INTGSBDM050N',
            10.0: 'INTGSBDM100N'
        },
        'GBP': {
            2.0: 'INTGSGBM020N',
            5.0: 'INTGSGBM050N',
            10.0: 'INTGSGBM100N'
        },
        'CHF': {
            2.0: 'INTGSCHM020N',
            5.0: 'INTGSCHM050N',
            10.0: 'INTGSCHM100N'
        }
    }

    # Sélection de la devise (défaut = USD si inconnue)
    points = rf_map.get(currency_code, rf_map['USD'])
    maturities = sorted(points.keys())
    rates = []

    print(f"--- Récupération Taux Sans Risque ({currency_code}) ---")

    for m in maturities:
        ticker = points[m]
        rate_value = None

        try:
            # CAS 1 : Ticker Yahoo (commence par ^)
            if ticker.startswith('^'):
                # fast_info est très rapide pour le dernier prix
                price = yf.Ticker(ticker).fast_info['last_price']
                if price is not None:
                    rate_value = price / 100.0  # Yahoo donne 4.5 pour 4.5%

            # CAS 2 : Code FRED (ne commence pas par ^)
            else:
                # FRED renvoie une série, on prend la dernière valeur dispo
                series = fred.get_series(ticker)
                if not series.empty:
                    rate_value = series.iloc[-1] / 100.0  # FRED donne aussi en %

            # Gestion d'erreur si la récupération échoue
            if rate_value is None or np.isnan(rate_value):
                raise ValueError("Pas de données")

            rates.append(rate_value)
            print(f"  Maturité {m}Y : {rate_value:.4f}")

        except Exception as e:
            print(f"  Erreur sur {ticker}: {e}")
            # Fallback : on reprend le dernier taux connu ou une valeur par défaut (3.5%)
            fallback = rates[-1] if rates else 0.035
            rates.append(fallback)

    # Interpolation ou Extrapolation linéaire
    rate_interp = np.interp(target_years, maturities, rates)

    print(f"  -> Taux interpolé pour {target_years:.2f} ans : {rate_interp:.4f}")
    return rate_interp


# --- ÉTAPE 2 : MODÈLES DE PRICING ---

def price_forward_equity(S0, risk_free, div_yield, T):
    return S0 * math.exp((risk_free - div_yield) * T)


def price_forward_currency(S0, r_domestic, r_foreign, T):
    """
    Calcule le prix Forward d'une devise (Interest Rate Parity).
    """
    return S0 * math.exp((r_domestic - r_foreign) * T)


def price_forward_fixed_income(S0, r, T, pv_coupons=0):
    """
    Calcule le prix Forward d'une obligation (Fixed Income).
    """
    return (S0 - pv_coupons) * math.exp(r * T)


def calculate_pv_coupons(coupon_rate, nominal, risk_free, forward_T, frequency=2):
    """
    Calcule la Present Value (PV) des coupons versés avant l'échéance du forward.
    """
    pv_total = 0
    coupon_amount = (coupon_rate * nominal) / frequency

    # On itère sur les dates de paiement possibles jusqu'à l'échéance du forward
    t = 1 / frequency
    while t <= forward_T:
        # Actualisation continue : CF * e^(-rt)
        pv_total += coupon_amount * math.exp(-risk_free * t)
        t += 1 / frequency

    return pv_total


# --- ÉTAPE 3 : VALORISATION LIVE ---

def value_forward_live(S0, K, risk_free, T):
    return S0 - (K * math.exp(-risk_free * T))


def value_fra_live(N, rfix, rfloat, days, position='long'):
    """
    Calcule le payoff d'un FRA payé au début de la période de prêt.
    """
    h = days / 360
    value_fra_t = N * ((rfloat - rfix) * h) / (1 + (rfloat * h))
    return value_fra_t if position.lower() == 'long' else -value_fra_t


# --- ÉTAPE 4 : DÉTECTION D'ARBITRAGE ---

def detect_arbitrage(theoretical_price, market_price, ticker_name, transaction_costs=0.0010):
    """
    Détecte les opportunités d'arbitrage.
    """
    diff = (market_price - theoretical_price) / theoretical_price
    print("-" * 50)
    print(f"ANALYSE D'ARBITRAGE POUR : {ticker_name}")
    print(f"Prix Théorique : {theoretical_price:.4f}")
    print(f"Prix Marché    : {market_price:.4f}")
    print(f"Écart Relatif  : {diff * 100:.3f}%")

    if diff > transaction_costs:
        print("\n[!] OPPORTUNITÉ DÉTECTÉE : ARBITRAGE 'CASH AND CARRY'")
        print("STRATÉGIE : Emprunter cash -> Acheter Spot -> Vendre Forward")
    elif diff < -transaction_costs:
        print("\n[!] OPPORTUNITÉ DÉTECTÉE : ARBITRAGE 'REVERSE CASH AND CARRY'")
        print("STRATÉGIE : Vendre à découvert Spot -> Placer cash -> Acheter Forward")
    else:
        print("\n[✓] AUCUN ARBITRAGE POSSIBLE (Écart trop faible)")
    print("-" * 50)


# --- SCRIPT PRINCIPAL ---

if __name__ == "__main__":
    print("=== QUANT PRICER - ANALYSE LIVE ===")

    # Inputs utilisateur
    ticker_input = input('Ticker (ex: AAPL, EURUSD=X, GC=F) : ').upper()
    if len(ticker_input) == 6:
        ticker_input = ticker_input + '=X'
    try:
        m_days = float(input('Maturité (en jours, ex: 90) : '))
    except ValueError:
        m_days = 90
        print("Maturité par défaut fixée à 90 jours.")

    # 1. Acquisition des données
    # Note: On unpack 5 valeurs comme défini dans get_market_data
    spot, q, ticker_obj, curr, asset_type = get_market_data(ticker_input)

    if spot and ticker_obj:
        T_years = m_days / 365
        # Récupération du taux sans risque domestique
        r_dom = get_rf_rate_by_currency(curr, m_days)

        print(f"\n--- Résultats Marché {ticker_input} ({asset_type}) ---")
        print(f"Prix Spot : {spot:.4f} {curr}")

        # 2. Choix du modèle de Pricing
        if asset_type == 'CURRENCY':
            # Extraction des codes devises
            base_curr = ticker_obj.ticker[0:3]
            quote_curr = ticker_obj.ticker[3:6]

            r_base = get_rf_rate_by_currency(base_curr, m_days)
            r_quote = get_rf_rate_by_currency(quote_curr, m_days)

            # Modèle IRP : F = S * exp((r_quote - r_base) * T)
            f_theo = price_forward_currency(spot, r_quote, r_base, T_years)

            print(f"Taux {base_curr} (Base) : {r_base * 100:.2f}%")
            print(f"Taux {quote_curr} (Quote) : {r_quote * 100:.2f}%")

        elif asset_type in ['BOND', 'FIXEDINCOME'] or ticker_input.startswith('^'):
            # Calcul précis de la PV des coupons pour Fixed Income
            coupon_rate = 0.04  # Exemple : 4% annuel
            nominal = 100  # Exemple : 100
            frequency = 2  # Exemple : Semestriel
            pv_c = calculate_pv_coupons(coupon_rate, nominal, r_dom, T_years, frequency)

            f_theo = price_forward_fixed_income(spot, r_dom, T_years, pv_coupons=pv_c)
            print(f"Taux Sans Risque : {r_dom * 100:.2f}%")
            print(f"PV Coupons (Calculé) : {pv_c:.4f} {curr}")

        else:
            # Modèle par défaut (Actions/ETF) utilisant le dividende continu q
            f_theo = price_forward_equity(spot, r_dom, q, T_years)
            print(f"Taux Sans Risque : {r_dom * 100:.2f}%")
            print(f"Dividend Yield (q) : {q * 100:.2f}%")

        print(f"PRIX FORWARD THÉORIQUE : {f_theo:.4f}")
