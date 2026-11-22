from pybit.unified_trading import HTTP

session = HTTP(
    testnet=False,
    api_key="XjTKcbuhJLiyleeIfK",
    api_secret="TERNG4X9TviYjKq8vVZ4e2YmRmaH91YoNswG",
)

# Get the orderbook of the USDT Perpetual, BTCUSDT
session.get_orderbook(category="linear", symbol="BTCUSDT")

# Create five long USDC Options orders.
# (Currently, only USDC Options support sending orders in bulk.)
payload = {"category": "option"}
orders = [{
  "symbol": "BTC-30JUN23-20000-C",
  "side": "Buy",
  "orderType": "Limit",
  "qty": "0.1",
  "price": i,
} for i in [15000, 15500, 16000, 16500, 16600]]

payload["request"] = orders
# Submit the orders in bulk.
session.place_batch_order(payload)