# Version 0.1.0

Initial study release containing the 60 diagnostic runs and 100 accuracy-comparison files underlying the three-panel figure, standalone training runners, randomized trace-kernel diagnostics, and reproducible analysis. Panel A compares model averaging global accuracy with individual local-only models on their own-band and global tests.

The reported mean retrospective lead is 529.5 steps, with 18/20 positive leads. Pooled training crosses the threshold in 17/20 runs, failing the planned federation-specificity test. Results remain preliminary.

See [README.md](README.md) for methods and limitations, and [VALIDATION.md](VALIDATION.md) for the checks performed on this release.

## Documentation clarification

Clarified the data-access difference between isolated clients and model averaging, and the distinction between exact reproduction and correctness. The figure, saved results, numerical statistics, training implementation, and verification scope are unchanged.

## Publication preparation

The manuscript is in preparation for submission; a draft is available upon request. Source formatting has been standardized, an unused test import removed, and plotting comments clarified. The lead analysis now rejects censored runs instead of silently treating a missing onset as a zero lead. The saved battery has no censored runs, so its results, figure, and statistics are unchanged; training behavior is unchanged.
