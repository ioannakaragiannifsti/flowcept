# Release rollback policy

A rollback is preferred over a forward fix when the faulty change is recent, isolated, and
reversible. Roll back one instance at a time and verify service health between instances.

Rollbacks that touch credential storage require confirming that the previous credential
source is still populated, otherwise the rolled-back version fails to start for the same
reason the new one did.
