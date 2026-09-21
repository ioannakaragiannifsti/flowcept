# Email worker configuration reference

The email worker reads its SMTP credentials once, at startup. There is no runtime reload:
an invalid value fails closed and the worker exits during configuration load rather than
degrading to a retry loop.

Credentials moved to the secrets store in release 4.12.0. Workers deployed before that
release read them from the legacy environment file, so a partial rollout can leave some
workers reading one source and some the other.
