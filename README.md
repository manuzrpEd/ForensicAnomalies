# LAPD Crime Data Anomaly Detection – Summary

This project uses both a **Variational Autoencoder (VAE)** and an **Isolation Forest** to detect anomalous crime records in LAPD data (~210,000 records, 5% contamination).  

## Key Findings

### Normal Crime Patterns
- Single-victim, single-offense incidents.
- Crimes against **property** (theft, burglary, vandalism).
- Victims typically **Hispanic adult males**, aged 30–45.
- Case status usually **Investigation Continued**.
- Weapon information often **missing**.
- Homeless involvement, transit-related crimes, domestic violence, or gang involvement are **rare**.

### Anomalous Crime Patterns
- **Multiple victims** or multiple offenses.
- Crimes involving **actual persons**, often female or non-typical demographics.
- Cases that are **Cleared by Arrest** rather than ongoing.
- **Homeless arrestee or victim involvement**.
- Transit-related crimes (bus/metro) or unusual locations.
- Rarely reported crime types or societal-level offenses.

### Insights
- Both the VAE and Isolation Forest **highlight the same unusual patterns**, confirming the presence of **high-impact, socially salient incidents**.
- These methods automatically surface crimes that, while statistically rare (~5%), are the ones that generate the most public and policy concern in Los Angeles.