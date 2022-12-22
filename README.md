# Snowflake-Spend-Forecaster
Linear Regression and Random Forest Regression ML models that can *somewhat* accurtely forecast future Snowflake Credit Usage

Run:
```
select * from snowflake.account_usage.warehouse_metering_history
where start_time > dateadd(year, -2, current_date());
```
in Snowflake and download the output as a CSV to obtain the necessary data for this forecaster.
