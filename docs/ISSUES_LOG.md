# Issues & Troubleshooting Log

This log tracks known issues, user-reported bugs, and observations from internal testing for ALchemist. It is maintained by the development team.

---

## How to Report an Issue

If you encounter a problem or have feedback, please [open an issue on GitHub](https://github.com/NREL/ALchemist/issues) or email [ccoatney@nrel.gov](mailto:ccoatney@nrel.gov) with the following information:

- **Brief description of the issue**
- **Steps to reproduce (if applicable)**
- **Your operating system and environment**
- **Any error messages or screenshots**
- **Date observed**

---

## Known Issues

| Issue                                                                                         | Date Reported | Status      | Notes / Workarounds                                                                                 |
|-----------------------------------------------------------------------------------------------|---------------|-------------|-----------------------------------------------------------------------------------------------------|
| BoTorch kernel hyperparameters not shown in "Next Point" dialog                               | 2024-06-16    | Open        | Hyperparameters are available in the console output. UI fix planned.                                |
| Loading variables from CSV does not work; only JSON loads correctly                           | 2025-06-29    | Open        | Workaround: Use the application to save variables as .json and load the .json file.                 |
| Saving variables as CSV and reloading does not restore variables                              | 2025-06-29    | Open        | Same as above; use .json format for now.                                                            |
| Model Prediction Optimum tool: suggested experiment gives fractional value for integer variable| 2025-06-29    | Open        | Needs investigation                                                   |
| Model Prediction Optimum tool: optimizing to maximum or minimum gives same suggested values   | 2025-06-29    | Open        | Needs investigation                          |

---

## Resolved Issues

| Issue                                                                 | Date Reported | Date Resolved | Notes                                                                                               |
|-----------------------------------------------------------------------|---------------|---------------|-----------------------------------------------------------------------------------------------------|
| GUI not displaying fully on macOS; windows may be cut off             | 2024-06-16    | 2025-06-29    | Resolved as of latest testing; GUI now displays correctly on Mac without external monitor.           |

---

This log is updated as issues are reported and resolved.