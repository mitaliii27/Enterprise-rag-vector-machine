# Invoice Verification

## Three-way match
Every invoice is matched against the purchase order and the goods receipt. The matching tolerance is 2% or EUR 50, whichever is lower. Invoices outside tolerance are blocked for payment.

## Missing information
Invoices without a purchase order number are returned to the supplier. Duplicate invoices are detected on the combination of supplier, invoice number and amount.

## Payment
The standard payment term is 30 days. An early payment discount of 2% if paid within 10 days is taken wherever the supplier offers it. Invoices above EUR 50,000 require a second approver.
