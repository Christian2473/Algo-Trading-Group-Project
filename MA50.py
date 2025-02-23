from AlgorithmImports import *
import pandas as pd
import numpy as np
from datetime import timedelta

class Rsi5mMultiTimeframeAlgorithm(QCAlgorithm):

    # ======= CHANGEABLE PARAMETERS =========

    # Trading asset(s)
    TICKERS = ["SPY"]
    
    # Candle settings (in minutes)
    CANDLE_MINUTES = 5
    
    # Warmup period (minutes)
    WARMUP_PERIOD_MINUTES = 2
    
    # Initial trade parameters (executed only once)
    INITIAL_BUY_QTY = -1000
    
    # RSI periods for the 5m bars
    RSI7_PERIOD = 7
    RSI14_PERIOD = 14
    RSI28_PERIOD = 28
    
    # Threshold for when RSI7 and RSI14 are "equal" (in /100 points)
    RSI_EQUAL_THRESHOLD = 4
    
    # Extra buy condition: if RSI7 is at least this many points below RSI28, add extra buy qty
    RSI_BUY_EXTRA_THRESHOLD = 12
    
    # Extra sell condition: if RSI7 is at least this many points above RSI28, increase sell qty
    RSI_SELL_EXTRA_THRESHOLD = 14
    
    # Base order sizes for trade candles
    BASE_BUY_QTY = 0
    BASE_SELL_QTY = 0

    # Addition order sizes (trigger when extra buy and sell condition met)
    QTY_5M_DOWN_BUY = 2
    QTY_RSI_CROSS_BUY = 10
    QTY_5M_UP_SELL = 2
    QTY_RSI_CROSS_SELL = 5

    # Maximum retry count for order placement failures
    MAX_ORDER_RETRIES = 1

    # Newly added MA50 parameters 
    MA50_PERIOD = 50            # number of bars for the MA50 calculation
    MA50_TARGET_POSITION = 100  # desired position size if price is above MA50

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2025, 2, 20)
        self.SetCash(1000000)
        
        # Add equities
        self.symbols = {}
        for ticker in self.TICKERS:
            equity = self.AddEquity(ticker, Resolution.Minute)
            self.symbols[ticker] = equity.Symbol
        self.spy = self.symbols["SPY"]

        # Set up a 5-minute consolidator for SPY
        self.consolidator = TradeBarConsolidator(timedelta(minutes=self.CANDLE_MINUTES))
        self.SubscriptionManager.AddConsolidator(self.spy, self.consolidator)
        self.consolidator.DataConsolidated += self.OnDataConsolidated

        # Create RSI indicators – these will be reinitialized at the start of each day
        self.rsi7 = RelativeStrengthIndex(self.RSI7_PERIOD, MovingAverageType.Simple)
        self.rsi14 = RelativeStrengthIndex(self.RSI14_PERIOD, MovingAverageType.Simple)
        self.rsi28 = RelativeStrengthIndex(self.RSI28_PERIOD, MovingAverageType.Simple)
        
        # Newly added MA50 indicator 
        self.ma50 = SimpleMovingAverage(self.MA50_PERIOD)  # create the MA50 indicator

        # For comparing current candle with previous one
        self.prevBar = None

        # Track warmup and order state
        self.warmupEndTime = None
        # initialTradeDone is for each day; hasDoneInitialTrade is global for the algorithm lifetime
        self.initialTradeDone = False  
        self.hasDoneInitialTrade = False
        
        # For dynamic buy orders on a given day, store the last buy quantity used
        self.lastBuyQty = self.BASE_BUY_QTY
        self.lastSellQty = self.BASE_SELL_QTY

        # Track fill prices for the last 5 buy orders
        self.recentBuyPrices = []  

        # To detect a new trading day to reinitialize RSI(s)
        self.currentTradingDay = None

        # Track the last signal direction ("Buy" or "Sell")
        self.lastSignalDirection = None

        # Order retry tracking: {orderTicketId: retryCount}
        self.orderRetryCounts = {}

        # Initialize logging counters for each category
        self.logCounts = {"INFO": 0, "SIGNAL": 0, "ORDER": 0, "ERROR": 0, "HISTORY": 0}

        self.LogAction("INFO", "Algorithm Initialized.")

    def OnDataConsolidated(self, sender, bar):
        try:
            currentTime = bar.EndTime

            # Check if a new trading day has begun:
            if self.currentTradingDay != currentTime.date():
                self.currentTradingDay = currentTime.date()
                self.ReinitializeIndicators(currentTime)
                self.lastBuyQty = self.BASE_BUY_QTY   # Reset dynamic buy quantity each day
                self.initialTradeDone = False         # Reset the initial trade flag for this day
                self.lastSignalDirection = None       # Reset signal direction at start of day
                self.LogAction("INFO", "New trading day: {}. Indicators reinitialized, order sizes and signal direction reset."
                               .format(currentTime.date()))
                # Set warmup end time for the new day
                self.warmupEndTime = currentTime + timedelta(minutes=self.WARMUP_PERIOD_MINUTES)
            
            # Update the RSI indicators with current bar price
            self.rsi7.Update(currentTime, bar.Close)
            self.rsi14.Update(currentTime, bar.Close)
            self.rsi28.Update(currentTime, bar.Close)

            # Update the MA50 with current bar price 
            self.ma50.Update(currentTime, bar.Close)  # new line
            
            # During warmup, just update the previous bar and log info
            if currentTime < self.warmupEndTime:
                self.prevBar = bar
                return

            # Once warmup is complete, perform the initial trade if not already done and if it's the first day only.
            if not self.initialTradeDone and not self.hasDoneInitialTrade:
                history = self.History(self.spy, 1, Resolution.Daily)
                if not history.empty:
                    previousClose = list(history["close"])[0]
                    if bar.Close < previousClose:
                        self.PlaceOrderWithRetry(self.spy, self.INITIAL_BUY_QTY, "Initial Buy")
                        self.LogAction("SIGNAL", "Initial trade executed: {} SPY (bar close {} vs previous close {})"
                                       .format(self.INITIAL_BUY_QTY, bar.Close, previousClose))
                    else:
                        self.LogAction("SIGNAL", "Initial trade skipped: bar close {} >= previous close {}"
                                       .format(bar.Close, previousClose))
                self.initialTradeDone = True
                self.hasDoneInitialTrade = True

            # Ensure the RSI indicators are ready (they might not be during the early minutes)
            if not (self.rsi7.IsReady and self.rsi14.IsReady and self.rsi28.IsReady):
                self.prevBar = bar
                return

            # --- Trading Logic ---
            rsi7Val = self.rsi7.Current.Value
            rsi14Val = self.rsi14.Current.Value
            rsi28Val = self.rsi28.Current.Value

            # 1. BUY conditions: both RSI7 and RSI14 are below RSI28.
            if (rsi7Val < rsi28Val) and (rsi14Val < rsi28Val):
                # Check for signal direction change (sell to buy) and reset dynamic buy size if necessary.
                if self.lastSignalDirection != "Buy":
                    if self.lastSignalDirection == "Sell":
                        self.lastBuyQty = self.BASE_BUY_QTY
                        self.LogAction("INFO", "Signal direction switched to BUY. Resetting dynamic buy quantity to {}."
                                    .format(self.BASE_BUY_QTY))
                    self.lastSignalDirection = "Buy"
                # Do not trade if RSI7 and RSI14 are too close
                if abs(rsi7Val - rsi14Val) <= self.RSI_EQUAL_THRESHOLD:
                    self.LogAction("SIGNAL", "Buy signal suppressed (RSI7: {:.2f}, RSI14: {:.2f} within threshold {})"
                                   .format(rsi7Val, rsi14Val, self.RSI_EQUAL_THRESHOLD))
                elif rsi7Val < rsi14Val:
                    qty = self.lastBuyQty  # starting with the base/dynamic quantity

                    # If current close is lower than previous candle's close, add +1
                    if self.prevBar is not None and bar.Close < self.prevBar.Close:
                        qty += self.QTY_5M_DOWN_BUY

                    # Additionally, if RSI7 is at least RSI_BUY_EXTRA_THRESHOLD points below RSI28, add +1
                    if rsi7Val <= rsi28Val - self.RSI_BUY_EXTRA_THRESHOLD:
                        qty += self.QTY_RSI_CROSS_BUY

                    self.PlaceOrderWithRetry(self.spy, qty, "Buy")
                    self.LogAction("SIGNAL", "Buy signal executed: {} SPY (RSI7: {:.2f}, RSI14: {:.2f}, RSI28: {:.2f}, Close: {})"
                                   .format(qty, rsi7Val, rsi14Val, rsi28Val, bar.Close))
                    # Update lastBuyQty for next candle
                    self.lastBuyQty = qty

            # 2. SELL conditions: both RSI7 and RSI14 are above RSI28.
            elif (rsi7Val > rsi28Val) and (rsi14Val > rsi28Val):
                # If we are switching from BUY to SELL, reset dynamic sell quantity only when last signal was BUY:
                if self.lastSignalDirection != "Sell":
                    if self.lastSignalDirection == "Buy":
                        self.lastSellQty = self.BASE_SELL_QTY
                        self.LogAction("INFO", "Switched from BUY to SELL. Dynamic sell quantity reset to {}."
                                       .format(self.BASE_SELL_QTY))
                    self.lastSignalDirection = "Sell"
                
                # Check exception: if any of the last 5 buy orders was filled at a higher price than current close, skip sell.
                if any(buyPrice > bar.Close for buyPrice in self.recentBuyPrices):
                    self.LogAction("SIGNAL", "Sell suppressed: At least one of the last 5 buy orders was at a higher price than the current close.")
                elif abs(rsi7Val - rsi14Val) <= self.RSI_EQUAL_THRESHOLD:
                    self.LogAction("SIGNAL", "Sell signal suppressed (RSI7: {:.2f}, RSI14: {:.2f} within threshold {})"
                                   .format(rsi7Val, rsi14Val, self.RSI_EQUAL_THRESHOLD))
                elif rsi7Val > rsi14Val:
                    qty = self.lastSellQty  # starting dynamic sell quantity
                    
                    if self.prevBar is not None and bar.Close > self.prevBar.Close:
                        qty += self.QTY_5M_UP_SELL
                        
                    if rsi7Val >= rsi28Val + self.RSI_SELL_EXTRA_THRESHOLD:
                        qty += self.QTY_RSI_CROSS_SELL

                    if self.Portfolio[self.spy].Quantity > 0:
                        self.PlaceOrderWithRetry(self.spy, -qty, "Sell")
                        self.LogAction("SIGNAL", "Sell signal executed: {} SPY (RSI7: {:.2f}, RSI14: {:.2f}, RSI28: {:.2f}, Close: {})"
                                       .format(qty, rsi7Val, rsi14Val, rsi28Val, bar.Close))
                        self.lastSellQty = qty  # update dynamic sell quantity
                    else:
                        self.LogAction("INFO", "Sell signal generated but no SPY position available.")

            # Update previous bar reference for next candle comparisons.
            self.prevBar = bar

            # Start of MA50 strategy logic 
            if self.ma50.IsReady:  # only proceed if MA50 has enough bars
                closePrice = bar.Close
                ma50Value = self.ma50.Current.Value

                # If the current price is above MA50, we want to hold a fixed position
                if closePrice > ma50Value:
                    currentHoldings = self.Portfolio[self.spy].Quantity
                    desiredHoldings = self.MA50_TARGET_POSITION
                    if currentHoldings < desiredHoldings:
                        qtyToBuy = desiredHoldings - currentHoldings
                        self.PlaceOrderWithRetry(self.spy, qtyToBuy, "MA50_Buy")
                        self.LogAction("SIGNAL",
                                       f"MA50_Buy signal: close {closePrice:.2f} > MA50 {ma50Value:.2f}. "
                                       f"Buy {qtyToBuy} shares to reach target {desiredHoldings}.")
                else:
                    # If the price is below MA50, we liquidate that particular long position
                    currentHoldings = self.Portfolio[self.spy].Quantity
                    if currentHoldings > 0:
                        self.PlaceOrderWithRetry(self.spy, -currentHoldings, "MA50_Sell")
                        self.LogAction("SIGNAL",
                                       f"MA50_Sell signal: close {closePrice:.2f} < MA50 {ma50Value:.2f}. "
                                       f"Sell all {currentHoldings} shares.")

        except Exception as e:
            self.LogAction("ERROR", "Exception in OnDataConsolidated: {}".format(e))

    def PlaceOrderWithRetry(self, symbol, quantity, orderType):
        """Places a market order with a retry mechanism if an error occurs."""
        try:
            orderTicket = self.MarketOrder(symbol, quantity)
            # Reset any retry counter for this order if successfully sent.
            self.orderRetryCounts[orderTicket.OrderId] = 0
            self.LogAction("ORDER", "{} order placed for {} shares of {}. OrderId: {}"
                           .format(orderType, quantity, symbol.Value, orderTicket.OrderId))
        except Exception as e:
            orderId = "Unknown"
            self.LogAction("ERROR", "{} order exception: {}. Will attempt to retry.".format(orderType, e))
            retries = self.orderRetryCounts.get(orderId, 0)
            if retries < self.MAX_ORDER_RETRIES:
                self.orderRetryCounts[orderId] = retries + 1
                self.PlaceOrderWithRetry(symbol, quantity, orderType)
            else:
                self.LogAction("ERROR", "{} order failed after {} retries.".format(orderType, retries))

    def ReinitializeIndicators(self, currentTime):
        """
        Reinitializes the RSI indicators at the start of a new trading day.
        This method also "seeds" the indicators with historical data.
        """
        self.rsi7.Reset()
        self.rsi14.Reset()
        self.rsi28.Reset()
        
        # Reset MA50 as well
        self.ma50.Reset()  # new line

        requiredBars = max(self.RSI28_PERIOD, self.MA50_PERIOD)  # ensure enough bars for RSI28 and MA50

        # Determine today's market open
        marketOpen = currentTime.replace(hour=9, minute=30, second=0, microsecond=0)
        
        # Get current day's minute history from market open until now.
        currentDayHistory = self.History(self.spy, marketOpen, currentTime, Resolution.Minute)
        currentDayBars = self.AggregateTo5Min(currentDayHistory)
        
        barsToFeed = list(currentDayBars)

        # If insufficient 5m bars, supplement with previous day's data.
        if len(barsToFeed) < requiredBars:
            extraNeeded = requiredBars - len(barsToFeed)
            previousDate = marketOpen.date() - timedelta(days=1)
            previousDayStart = marketOpen.replace(
                year=previousDate.year, month=previousDate.month, day=previousDate.day
            )
            previousDayEnd = previousDayStart + timedelta(hours=6, minutes=30)  # approximate trading hours
            previousHistory = self.History(self.spy, previousDayStart, previousDayEnd, Resolution.Minute)
            previousDayBars = self.AggregateTo5Min(previousHistory)
            extraBars = list(previousDayBars)[-extraNeeded:] if len(previousDayBars) >= extraNeeded else list(previousDayBars)
            barsToFeed = extraBars + barsToFeed

        # Feed bars into indicators.
        for bar in barsToFeed:
            self.rsi7.Update(bar.EndTime, bar.Close)
            self.rsi14.Update(bar.EndTime, bar.Close)
            self.rsi28.Update(bar.EndTime, bar.Close)
            
            # Also feed MA50 
            self.ma50.Update(bar.EndTime, bar.Close)  # new line
        
        self.LogAction("HISTORY", "Indicators seeded with {} bars ({} from previous day, {} from current day)."
                       .format(len(barsToFeed), max(0, requiredBars - len(currentDayBars)), len(currentDayBars)))
    
    def AggregateTo5Min(self, history):
        """
        Aggregates minute-level history data into 5-minute candles.
        
        Explanation:
        The History() call returns a DataFrame at minute resolution with a multi-index [symbol, time].  
        This helper method resamples the 'close' column into 5‑minute intervals (along with open, high, and low)  
        to simulate TradeBar objects. These aggregated bars are then used to seed RSI indicators.
        """
        if history.empty:
            return []
        
        df = history.loc[self.spy].copy()
        df.index = pd.to_datetime(df.index)
        ohlc = df['close'].resample('{}T'.format(self.CANDLE_MINUTES)).ohlc()
        bars = []
        for time, row in ohlc.iterrows():
            barObj = TradeBar(time, self.spy, row['open'], row['high'], row['low'], row['close'], 0)
            bars.append(barObj)
        return bars

    def OnOrderEvent(self, orderEvent):
        if orderEvent.Status == OrderStatus.Filled:
            # If a buy order is filled, record its fill price
            if orderEvent.FillQuantity > 0:
                self.recentBuyPrices.append(orderEvent.FillPrice)
                # Keep only the last 10 buy prices
                if len(self.recentBuyPrices) > 10:
                    self.recentBuyPrices.pop(0)
            self.LogAction("ORDER", "Order Filled: {} shares of {} at {}".format(
                orderEvent.FillQuantity, orderEvent.Symbol.Value, orderEvent.FillPrice))
        elif orderEvent.Status == OrderStatus.PartiallyFilled:
            self.LogAction("ORDER", "Order Partially Filled: {} shares of {} at {}".format(
                orderEvent.FillQuantity, orderEvent.Symbol.Value, orderEvent.FillPrice))
        elif orderEvent.Status == OrderStatus.Canceled:
            self.LogAction("ORDER", "Order Canceled: OrderId {}".format(orderEvent.OrderId))
        elif orderEvent.Status == OrderStatus.Invalid:
            self.LogAction("ERROR", "Order Invalid: OrderId {}".format(orderEvent.OrderId))

    def LogAction(self, category, message):
        """
        Helper function to log messages with a category.
        Categories include:
         - INFO: General information about algo state.
         - SIGNAL: Generated trading signals.
         - ORDER: Order placement and order events.
         - ERROR: Errors and exceptions.
         - HISTORY: Events related to history/data aggregation.
        """
        self.logCounts[category] += 1
        self.Debug("[{} {}] {}".format(category, self.logCounts[category], message))

    def OnEndOfAlgorithm(self):
        self.LogAction("INFO", "Algorithm finished. Final Portfolio Value: {}".format(self.Portfolio.TotalPortfolioValue))
