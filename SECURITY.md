# Security Policy

## Reporting a vulnerability

If you find a security vulnerability in Harmonic, please report it privately.

**Do not** open a public GitHub issue for it. Email **contact@khalm.ai** with:

- a description of the vulnerability;
- steps to reproduce it;
- what it could affect;
- a suggested fix, if you have one.

We acknowledge reports within 48 hours and tell you when to expect a fix.

## Scope

This policy covers:

- the API and its two modes, the operator console and the public try page
  (`api/`, `web/`);
- the knowledge-graph and diagnosis layers (`kg/`, `rag/`);
- the deploy scripts (`deploy/`).

It does not cover third-party services or tools the project uses.

## How the system is built to fail safely

1. **Advisory only.** Harmonic has no control path. It never drives an
   actuator; the plant's own safety system stays in charge.
2. **No dynamic allocation after init.** All of the engine's mutable state
   lives in a memory block the caller supplies.
3. **No floating point in the tick path.** Fixed-point (Q16.16) only, so
   results are identical on every platform.
4. **The public try page is isolated.** It runs as its own service with its own
   storage and service account, serves only the try flow, calls no AI model,
   and deletes uploads after 7 days unless the visitor chooses to keep them.
5. **Grounded answers.** A diagnosis may cite only manual sections that were
   retrieved; an invented citation is rejected.
