# Snowflake-Spend-Forecaster
Linear Regression and Random Forest Regression ML models that can *somewhat* accurately forecast future Warehouse Credit Usage in Snowflake

Run:
```
select * from snowflake.account_usage.warehouse_metering_history
where start_time > dateadd(year, -2, current_date());
```
in Snowflake and download the output as a CSV to obtain the necessary data for this forecaster.

The random forest regression model will likely take a few to ten minutes to run, depending on the params inputted. I'd recommend not going over k_folds > 5 and n_iter > 150 if you want your computer to survive.
