from ib_insync import *
import pandas as pd
import time


def get_minute_bar(contract):
    """
    Retrieve historical minute bar data for the given contract.
    This function requests the last 2 days of 1-minute interval trade data.
    """
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='2 D',  # Request 2 days of data
        barSizeSetting='1 min',  # Set bar size to 1 minute
        whatToShow='TRADES',  # Retrieve trade prices
        useRTH=True,  # Use regular trading hours only
        formatDate=1
    )

    # Define an empty DataFrame in case no data is returned
    df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    # Convert the retrieved data to a DataFrame
    if bars:
        df = pd.DataFrame(bars)
        df['date'] = pd.to_datetime(df['date'])  # Convert date to proper format

    return df


def calculate_rumi(df, short_window=10, long_window=50, period=20):
    """
    Compute the RUMI indicator and its first-order difference.
    The RUMI indicator is based on the difference between short-term and long-term SMAs.

    :param df: DataFrame containing historical price data.
    :param short_window: Window size for short-term SMA.
    :param long_window: Window size for long-term SMA.
    :param period: Window size for the rolling mean of SMA difference.
    :return: DataFrame with additional columns for SMA, RUMI, and RUMI first-order difference.
    """
    df = df.copy()  # Copy the original DataFrame to avoid modifying it directly

    # Compute simple moving averages (SMA)
    df['SMA_short'] = df['close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['close'].rolling(window=long_window).mean()

    # Compute the difference between short and long SMAs
    df['Diff'] = df['SMA_short'] - df['SMA_long']

    # Compute the RUMI indicator as the rolling mean of the difference
    df['RUMI'] = df['Diff'].rolling(window=period).mean()

    # Compute the first-order difference of RUMI (RUMI_Diff1)
    df['RUMI_Diff1'] = df['RUMI'].diff()

    return df


def trade_monitor(trade):
    """
    Monitor the trade status until it is either Filled, Cancelled, or Rejected.
    This function waits for the IB API to update the trade status and prints relevant execution details.

    :param trade: The trade order object.
    """
    ib.waitOnUpdate(timeout=10)
    print(f"🔄 Most Recent Order Status: {trade.orderStatus.status}")

    if trade.orderStatus.status == 'Filled':
        ib.sleep(2)  # Short pause before fetching fill details
        fill_details = trade.fills
        execution_id = fill_details[0].execution.execId
        execution_time = fill_details[0].execution.time
        execution_price = trade.orderStatus.avgFillPrice
        filled_qty = trade.orderStatus.filled
        execution_commission = fill_details[0].commissionReport.commission

        print(f"✅ Execution ID: {execution_id}")
        print(f"✅ Trade Time: {execution_time}")
        print(f"✅ Execution Price: {execution_price}")
        print(f"✅ Execution Quantity: {filled_qty}")
        print(f"✅ Commission for the last trade: {execution_commission}")

    time.sleep(2)  # Short delay before the next action


def close_position(action, quantity):
    """
    Close an existing position before opening a new one.

    :param action: "BUY" or "SELL" to close the position.
    :param quantity: Number of shares/contracts to close.
    """
    order = MarketOrder(action, quantity)
    trade = ib.placeOrder(contract, order)
    trade_monitor(trade)


def trade_rumi(df, symbol):
    """
    Execute trades based on the RUMI indicator's first-order difference.

    :param df: DataFrame containing calculated RUMI values.
    :param symbol: Stock symbol being traded.
    """
    latest_rumi = df['RUMI_Diff1'].iloc[-1]  # Get the latest RUMI_Diff1 value

    # Retrieve current position size for the given stock
    positions = ib.positions()
    position_size = 0
    for pos in positions:
        if pos.contract.symbol == symbol:
            position_size = pos.position

    # If a position exists, close it before opening a new one
    if position_size > 0 > latest_rumi:
        close_position("SELL", position_size)
    elif position_size < 0 < latest_rumi:
        close_position("BUY", abs(position_size))

    for pos in positions:
        if pos.contract.symbol == symbol:
            position_size = pos.position

    # Execute new trade based on RUMI condition
    if position_size == 0 and latest_rumi > 0:
        order = MarketOrder("BUY", 10)  # Buy 10 shares
        trade = ib.placeOrder(contract, order)
        print(f"🟢 Triggered Buy Order: RUMI={latest_rumi}")
        trade_monitor(trade)

    elif position_size == 0 and latest_rumi < 0:
        order = MarketOrder("SELL", 10)  # Sell 10 shares
        trade = ib.placeOrder(contract, order)
        print(f"🔴 Triggered Sell Order: RUMI={latest_rumi}")
        trade_monitor(trade)

    else:
        print(f"⚪ No Trade: RUMI={latest_rumi}")


if __name__ == '__main__':
    """
    Main execution block:
    - Connect to IBKR
    - Fetch historical data
    - Compute RUMI values
    - Execute trades based on RUMI signals
    - Disconnect after execution
    """
    ib = IB()
    ib.connect()

    Symbol = 'AAPL'
    contract = Stock(Symbol, "SMART", "USD")

    # Fetch minute-level historical data
    data = get_minute_bar(contract)

    # Extract relevant columns and set date as index
    DF = data[['date', 'close']].set_index('date')

    # Calculate RUMI indicator and execute trading strategy
    trade_rumi(calculate_rumi(DF), Symbol)

    # Disconnect from IBKR
    ib.disconnect()
