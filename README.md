# IIT_B-and-Mickinsey-

Problem Statement

KHAJANA Bank is worried about cost of their ATM operations. Can you help the bank to reduce the cost?

You are given a historical data for ATM daily cash withdrawals (training period). Not all data was captured successfully. And hence there are gaps in the dataset.

You are expected to:

    accurately forecast withdrawals for the test period
    Set replenishment policy for each ATM so that total cost of replenishment is minimum


Costs Involved
The cost of ATM replenishment involves 3 components

    Cost of refill: Refilling ATMs involves cost, for example, transportation/labor etc.
    Cost of cash: Banks earn interest on money through lending. Any money kept in ATMs is not available for lending and Banks looses interests on it. So, over stocking ATM leads to loss of revenue for Bank. If X% is the interest rate annually and if an ATM has funds on daily basis as 1000, 500, 300, 1000, etc. then cost of cash for that ATM for test month is calculated as (Sum (funds on each day) / 31 ) * (15%/12)
    Stock out cost = If an ATM runs out of cash there is a penalty!


Cost 	Value (INR)
Cost of refill 	300
Cost of Cash 	15% Annually
Cost of stock out 	1000 per day


Replenishment process:

    Replenishment happens at the beginning of the day
    If there is cash in the ATM at the time of replenishment then it is removed and a new box with cash equivalent to replenishment amount is inserted in ATM
    Stock is calculated at the end of day

You can use one of the replenishment strategy as follows:
Strategy 	Action
0 	do not replenish
1 	7 days per week (everyday replenish)
2 	replenish alternate days (day-1, day-3, day-5, etc.)
3 	Replenish two specific days per week - Thursday-Monday
4 	Replenish once weekly on every Thursday
5 	Replenish once weekly on every Monday
6 	Replenish once alternate week on Thursdays

https://static.analyticsvidhya.com/wp-content/uploads/2017/02/24144759/Image18.png
