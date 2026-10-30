# Security Policy

## Supported versions

This project ships a single line of development. Fixes go to `main` and to the
latest tag; older tags are not patched.

| Version | Supported |
|---|---|
| 0.1.x | Yes |

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Use GitHub's private reporting flow: open the **Security** tab of this repository
and choose **Report a vulnerability**. If that is unavailable, contact the
maintainer through the GitHub profile at https://github.com/2jason986.

Include the affected module, a reproduction or proof of concept, and the impact
you believe is reachable. You can expect an initial response within five working
days.

## Scope

The interesting attack surfaces in an agent pipeline are not the usual ones, so
this list is explicit:

- **Prompt injection through retrieved content.** Anything returned by a tool ends
  up in a model's context. A knowledge-base document that says "ignore previous
  instructions" is executed as data today; treat it as untrusted input.
- **Tool argument handling.** `ToolRegistry.call` never raises into the model, but
  it does not validate argument types either. A tool that shells out is a remote
  code execution hole.
- **Secret handling.** `ORCHESTRA_API_KEY` is read from the environment and never
  logged. Traces record tool arguments and model output; make sure they do not
  capture credentials in your own deployments.
- **Output trust.** Agent output is parsed as JSON and routed on, not executed.

## Out of scope

- Denial of service through token exhaustion on a paid model endpoint: that is a
  cost control problem, not a vulnerability in this library.
- Behaviour differences between model vendors.
