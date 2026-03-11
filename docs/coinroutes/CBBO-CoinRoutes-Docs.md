Streaming of a real time book and terminal price at available venues for a given currency pair at a user-defined quantity.

```
wss://<YOUR INSTANCE URL>/api/streaming/cbbo/
```

Authenticate the request by using the following header when connecting.

```
{ "Authorization" : "Token <YOUR TOKEN>" }
```

Once connected, to subscribe to data, you must send a json message with the following parameters

```
{
    "currency_pair":"BTC-USDT",
    "size_filter" : 0.00,
    "sample": 5,
    "subscribe": true
}
```

```
{
    "product": "BTC-USDT",
    "size_threshold": "0.00",
    "depth_limit": 1000,
    "generated_timestamp": "2025-06-03T13:26:00.008729",
    "processed_timestamp": "2025-06-03T13:26:00.008729",
    "markets": [
        "gemini",
        "bybit",
        "binance",
        "kucoin",
        "bitget",
        "okex",
        "gdax",
        "gateio",
        "mexc",
        "crypto_com",
        "deribit",
        "huobipro",
        "bitstamp",
        "binanceus",
        "m2",
        "kraken",
        "bullish",
        "bitfinex",
        "galaxy"
    ],
    "bids": [
        {
            "exchange": "okex",
            "level": 0,
            "price": "105459.80",
            "qty": "0.00814255",
            "total_qty": "0.00814255"
        },
        {
            "exchange": "gateio",
            "level": 0,
            "price": "105458.10",
            "qty": "0.15078",
            "total_qty": "0.15078"
        },
        {
            "exchange": "binance",
            "level": 0,
            "price": "105457.02",
            "qty": "0.01746",
            "total_qty": "0.03492000"
        },
        {
            "exchange": "m2",
            "level": 0,
            "price": "105457.02",
            "qty": "0.01746",
            "total_qty": "0.03492000"
        },
        {
            "exchange": "kraken",
            "level": 0,
            "price": "105453.30",
            "qty": "0.72395808",
            "total_qty": "0.72395808"
        },
        {
            "exchange": "mexc",
            "level": 0,
            "price": "105452.06",
            "qty": "0.774615",
            "total_qty": "0.774615"
        },
        {
            "exchange": "huobipro",
            "level": 0,
            "price": "105451.00",
            "qty": "0.20011",
            "total_qty": "0.20011"
        },
        {
            "exchange": "bitfinex",
            "level": 0,
            "price": "105450.00",
            "qty": "0.02034636",
            "total_qty": "0.02034636"
        },
        {
            "exchange": "kucoin",
            "level": 0,
            "price": "105449.90",
            "qty": "0.09080585",
            "total_qty": "0.09080585"
        },
        {
            "exchange": "crypto_com",
            "level": 0,
            "price": "105449.01",
            "qty": "0.0087",
            "total_qty": "0.0087"
        },
        {
            "exchange": "bybit",
            "level": 0,
            "price": "105446.70",
            "qty": "0.806569",
            "total_qty": "0.806569"
        },
        {
            "exchange": "bullish",
            "level": 0,
            "price": "105446.60",
            "qty": "0.00005534",
            "total_qty": "0.00005534"
        },
        {
            "exchange": "bitget",
            "level": 0,
            "price": "105439.99",
            "qty": "0.319249",
            "total_qty": "0.319249"
        },
        {
            "exchange": "gdax",
            "level": 0,
            "price": "105437.87",
            "qty": "0.01298344",
            "total_qty": "0.01298344"
        },
        {
            "exchange": "gemini",
            "level": 0,
            "price": "105435.50",
            "qty": "0.10235878",
            "total_qty": "0.10235878"
        },
        {
            "exchange": "bitstamp",
            "level": 0,
            "price": "105435.00",
            "qty": "0.01147833",
            "total_qty": "0.01147833"
        },
        {
            "exchange": "deribit",
            "level": 0,
            "price": "105422.00",
            "qty": "0.0316",
            "total_qty": "0.0316"
        },
        {
            "exchange": "binanceus",
            "level": 0,
            "price": "105360.82",
            "qty": "0.06381",
            "total_qty": "0.06381"
        },
        {
            "exchange": "galaxy",
            "level": 0,
            "price": "101102.10",
            "qty": "0.07",
            "total_qty": "0.07"
        }
    ],
    "asks": [
        {
            "exchange": "bitget",
            "level": 0,
            "price": "105440.00",
            "qty": "0.370803",
            "total_qty": "0.370803"
        },
        {
            "exchange": "bybit",
            "level": 0,
            "price": "105446.80",
            "qty": "1.826173",
            "total_qty": "1.826173"
        },
        {
            "exchange": "binanceus",
            "level": 0,
            "price": "105447.33",
            "qty": "0.04215",
            "total_qty": "0.04215"
        },
        {
            "exchange": "crypto_com",
            "level": 0,
            "price": "105449.02",
            "qty": "0.7482",
            "total_qty": "0.7482"
        },
        {
            "exchange": "kucoin",
            "level": 0,
            "price": "105450.00",
            "qty": "1.50851561",
            "total_qty": "1.50851561"
        },
        {
            "exchange": "huobipro",
            "level": 0,
            "price": "105451.01",
            "qty": "1.255357",
            "total_qty": "1.255357"
        },
        {
            "exchange": "gdax",
            "level": 0,
            "price": "105451.43",
            "qty": "0.02163843",
            "total_qty": "0.02163843"
        },
        {
            "exchange": "bullish",
            "level": 0,
            "price": "105451.50",
            "qty": "0.04324694",
            "total_qty": "0.04324694"
        },
        {
            "exchange": "mexc",
            "level": 0,
            "price": "105452.07",
            "qty": "16.603245",
            "total_qty": "16.603245"
        },
        {
            "exchange": "kraken",
            "level": 0,
            "price": "105453.40",
            "qty": "0.030355",
            "total_qty": "0.030355"
        },
        {
            "exchange": "binance",
            "level": 0,
            "price": "105457.03",
            "qty": "11.2994",
            "total_qty": "22.59880"
        },
        {
            "exchange": "m2",
            "level": 0,
            "price": "105457.03",
            "qty": "11.2994",
            "total_qty": "22.59880"
        },
        {
            "exchange": "bitstamp",
            "level": 0,
            "price": "105458.00",
            "qty": "0.002",
            "total_qty": "0.002"
        },
        {
            "exchange": "gateio",
            "level": 0,
            "price": "105458.20",
            "qty": "0.53514",
            "total_qty": "0.53514"
        },
        {
            "exchange": "okex",
            "level": 0,
            "price": "105459.90",
            "qty": "0.67461437",
            "total_qty": "0.67461437"
        },
        {
            "exchange": "bitfinex",
            "level": 0,
            "price": "105460.00",
            "qty": "0.37992378",
            "total_qty": "0.37992378"
        },
        {
            "exchange": "gemini",
            "level": 0,
            "price": "105472.26",
            "qty": "0.0006",
            "total_qty": "0.0006"
        },
        {
            "exchange": "deribit",
            "level": 0,
            "price": "105481.00",
            "qty": "0.1933",
            "total_qty": "0.1933"
        },
        {
            "exchange": "galaxy",
            "level": 0,
            "price": "105521.39",
            "qty": "0.07",
            "total_qty": "0.07"
        }
    ],
    "properties": {},
    "last_rx_nanos": 1748957158832374915,
    "internal_latency_nanos": 1176704174,
    "last_source_nanos": 1748957159000000000
}
```