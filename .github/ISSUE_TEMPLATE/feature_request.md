---
name: Feature request
about: Suggest a new proposal, sampler capability, or API improvement
labels: enhancement
---

## What problem are you trying to solve?

Describe the analysis you're doing and what's missing or awkward
(e.g. "I need a custom jump for correlated parameters", "I want to resume
runs across machines").

## Proposed solution

What you'd like impulse-mcmc to do. Sketch the API if you have one in mind.

## Alternatives

Workarounds you've tried (custom proposals via `add_custom_jump`, other
packages, ...) and why they fall short.

## Notes

If this involves a new jump proposal: proposals must return `(sample, qxy)`
with a correct Hastings ratio — see CONTRIBUTING.md if you plan to implement
it yourself.
