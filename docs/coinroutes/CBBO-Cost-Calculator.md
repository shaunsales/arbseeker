## [](https://uatj.coinroutes.io/doc/apis/merx/cost_calculator.html#merx---cost-calculator-rest---get-request)MERX - Cost Calculator (REST - GET Request)

Gives you real time buy/sell liquidity for a given currency pair across selected venue(s) for a specified quantity.

```
https://<YOUR INSTANCE URL>/api/cost_calculator/
```

Authenticate the request by using the following header.

```
{ "Authorization" : "Token <YOUR TOKEN>" }
```

#### [](https://uatj.coinroutes.io/doc/apis/merx/cost_calculator.html#cost-calculator-path-parameters)Cost Calculator Path Parameters

|Name|Type|Description|Required|
|---|---|---|---|
|symbol|string|The trading pair symbol (e.g., BTC-USDT).|Yes|

#### [](https://uatj.coinroutes.io/doc/apis/merx/cost_calculator.html#cost-calculator-query-parameters)Cost Calculator Query Parameters

|Name|Type|Description|Required|Example|
|---|---|---|---|---|
|side|string|Order direction, must be one of buy, sell, or both.|Yes|side=buy|
|target\_quantity|float|Specifies the target quantity to trade.|No - _(One of target\_quantity, funding\_quantity, or inverse\_target\_quantity is required)_|target\_quantity=1.5|
|funding\_quantity|float|Specifies the funding quantity instead of target quantity.|No - _(One of target\_quantity, funding\_quantity, or inverse\_target\_quantity is required)_|funding\_quantity=1000|
|inverse\_target\_quantity|float|Specifies an inverse target quantity.|No - _(One of target\_quantity, funding\_quantity, or inverse\_target\_quantity is required)_|inverse\_target\_quantity=2|
|markets|string|A semicolon-separated list of markets to consider. Each market can optionally specify a limit using :.|No|markets=BINANCE;COINBASE:50000|
|client|string|The client identifier used to fetch a specific book instrument.|No|client=123|
|fees-{market}|float|Specifies a fee percentage for a specific market.|No|fees-BINANCE=0.001|
|fees-{client\_id}-{market}|float|Specifies a client-specific fee for a market. Only applicable if a client\_id is determined from markets or explicitly passed via client.|No|fees-BINANCE=0.001|
|depth\_limit|integer|Maximum depth of the order book to consider.|No|depth\_limit=10|
|price\_limit|float|Sets an absolute price limit.|No|price\_limit=50000|
|price\_limit\_percent|float|Sets a percentage-based price limit. Cannot be used with price\_limit.|No|price\_limit\_percent=1.5|
|omit\_matches|boolean|Whether to omit matching orders. Must be true or false.|No|omit\_matches=true|
|quantity\_precision|string or integer|Specifies precision for quantity calculations. Use auto to fetch max precision.|No|quantity\_precision=auto|

#### [](https://uatj.coinroutes.io/doc/apis/merx/cost_calculator.html#cost-calculator-response)Cost Calculator Response

|Field|Type|Details|
|---|---|---|
|product|string|The product (or currency pair) that the data is for|
|gross\_consideration|string|Gross consideration of the order.|
|net\_consideration|string|Gross consideration of the order.|
|fees|string|Fees for the order.|
|quantity|string|Order quantity.|
|pricelevel\_count|integer|Number of matches in the book.|
|first\_price|string|First price on the order book.|
|last\_price|string|Last price on the order book.|

### [](https://uatj.coinroutes.io/doc/apis/merx/cost_calculator.html#example)Example

#### [](https://uatj.coinroutes.io/doc/apis/merx/cost_calculator.html#query)Query

```
curl -X GET "https://yourcompany.coinroutes.io/api/cost_calculator/BTC-USDT?side=buy&target_quantity=1.5&markets=BINANCE&fees-BINANCE=0.001&depth_limit=10" \
     -H "Authorization: Token YOUR_ACCESS_TOKEN"
```

#### [](https://uatj.coinroutes.io/doc/apis/merx/cost_calculator.html#response)Response

```
{
        "side": "buy",
        "markets": "BINANCE",
        "target_quantity": "1.5",
        "funding_quantity": null,
        "inverse_target_quantity": null,
        "sweep_orders": {
            "BINANCE": [
                "97742.40",
                "1.5"
            ]
        },
        "totals": {
            "market": {
                "BINANCE": {
                    "gross_consideration": "146613.600",
                    "net_consideration": "146615.0661360",
                    "fees": "1.4661360",
                    "quantity": "1.5",
                    "pricelevel_count": 1,
                    "first_price": "97742.40",
                    "last_price": "97742.40"
                }
            },
            "all": {
                "gross_consideration": "146613.600",
                "net_consideration": "146615.0661360",
                "fees": "1.4661360",
                "quantity": "1.5",
                "pricelevel_count": 1,
                "first_price": "97742.40",
                "last_price": "97742.40"
            }
        },
        "matches": [
            [
                "97743.377424",
                "BINANCE",
                "97742.40",
                "1.5"
            ]
        ],
        "last_rx_nanos": 1740070209016120607,
        "last_rx_nanos_market": "BINANCE",
        "internal_latency_nanos": 980224453,
        "last_source_nanos": 1740070209014000000,
        "last_source_nanos_market": "BINANCE",
        "fast": "yes"
    }
```

___