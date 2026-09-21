# Payment gateway failover

Fail over to the secondary payment gateway when authorization latency exceeds two seconds
or the success rate falls below 98 percent. Failover interrupts in-flight authorizations,
so it is only justified when payments are themselves degraded.

Do not fail over payments in response to incidents in unrelated services.
