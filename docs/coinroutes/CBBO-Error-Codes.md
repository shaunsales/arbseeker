## [](https://uatj.coinroutes.io/doc/apis/merx/error-codes.html#merx---market-data)MERX - Market Data

Where possible, Merx will respond with an error to describe why a request was incorrect or has failed.

The error response is a json object in the following form:

```
{
    "type": "error",
    "error_code": 123456,
    "error": "<ERROR NAME>",
    "error_text": "<Description of error>",
}
```

|Field Name|Type|Description|
|---|---|---|
|type|String|All error messages will include this type parameter|
|error\_code|Number|Error code as per table of errors below|
|error\_name|String|Human readable error name as per table of errors below|
|error\_description|String|Description of the error, or instruction to rectify request. This may be a generic description or may be specific to the request|

There may be instances, where when possible, Merx will respond with an error description to aid the client in rectifying the response. As an example below, when attempting to subscribe to cbbo with an invalid size filter, Merx will list all valid size filters.

```
{
    "type": "error",
    "error_code": 100003,
    "error": "INVALID_SIZE_FILTER",
    "error_text": "0.23 is an invalid size filter for BTC-USDT. 
                   Available size filters are 
                   [0.0, 0.2, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]"
}
```

## [](https://uatj.coinroutes.io/doc/apis/merx/error-codes.html#table-of-errors)Table of Errors

|Error Name|Error Code|Description|
|---|---|---|
|INVALID\_CURRENCY\_PAIR|100001|Please check currency pair|
|INVALID\_SUBSCRIPTION\_MESSAGE|100002|Unable to parse subscription message|
|INVALID\_SIZE\_FILTER|100003|Please check size filter is valid|
|INVALID\_DEPTH\_LIMIT|100004|Please check depth limit is valid|
|INVALID\_EXCHANGES|100005|Please check exchanges are valid|
|INVALID\_REQUEST|100006|Unable to get data. Please check request parameters|
|RESPONSE\_ERROR|101001|Unable to generate error response|
|INVALID\_TOKEN|101002|Invalid Token. Please try again with a valid token|
|AUTH\_SERVICE\_UNAVAILABLE|101003|Authentication service unavailable. Please try again later|
|AUTH\_INTERNAL\_ERROR|101004|Authentication service internal error. Please try again later|
|AWAITING\_SYMBOL\_DATA|101005|Server is initializing and awaiting metadata. Please try again later|
|ALREADY\_SUBSCRIBED|101006|Already subscribed to this subscription|
|SERVER\_ERROR|101007|Unexpected Server Error|

___