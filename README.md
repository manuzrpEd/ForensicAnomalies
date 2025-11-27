# LAPD Crime Data Anomaly Detection – Summary

This project uses both a **Variational Autoencoder (VAE)** and an **Isolation Forest** to detect anomalous crime records in LAPD data (~210,000 records, 5% contamination).  

A **Variational Autoencoder (VAE)** is a type of neural network used for unsupervised learning, especially for dimensionality reduction, generative modeling, and anomaly detection. It is a type of autoencoder, which is a network that learns to compress data into a smaller representation (encoding) and then reconstruct it (decoding).

The “variational” part means it doesn’t just encode data into fixed numbers—it encodes each input as a probability distribution (usually a Gaussian with a mean and standard deviation).

- Encoder: Turns input data into a distribution over a latent space (mean + variance).
- Latent space: A compressed, continuous representation of the data.
- Decoder: Samples from the latent distribution and reconstructs the input.

An **Isolation Forest** is a machine learning algorithm designed specifically for anomaly or outlier detection.

1. Randomly select a feature and a split value between the min and max of that feature.
2. Split the data into two branches based on that value.
3. Repeat recursively until each point is isolated or a maximum tree depth is reached.
4. Anomaly score:
    - Points that get isolated quickly (in fewer splits) have a high anomaly score.
    - Points that take many splits are likely normal.

- The “forest” part comes from averaging across many random trees to get a stable anomaly score.

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