# Global Horizontal Irradiance (GHI) Nowcasting and Short-Term Forecasting Using the Folsom Dataset.
This project is my undergraduate thesis at Universidade Federal de Santa Catarina (UFSC) for my Electrical Engineering degree, and it is dedicated to my friends, family and the professors that guided me throughout the course.

---
## Motivation
As technology advances and our electrical devices become more and more powerful, the demand for electric energy increases. Our job, as engineers, is to supply this ever-increasing demand in a safe, sustainable and renewable way. It is a well known fact that solar photovoltaic energy is both sustainable and renewable, however, it's large-scale integration into the electric system still represents a safety concern to the grid operators around the world. This is mainly due to the highly unpredictable and fast-changing movement of the clouds that continuously block and unblock the sun from the solar farms, causing it's power output to be greatly irregular. If this irregularity is not accounted for by the grid operator there will be an inbalance between the supply and demand, which can cause the frequency/voltage of the system to exceed it's tolerated limits. Depending on how much these limits where exceeded by, the grid may suffer catastrophic mechanical and eletrical consequences.
Therefore, the goal of this project is to train deep learning models to nowcast and forecast the GHI 5 minutes ahead of time. To do this, the Folsom dataset (
https://doi.org/10.1063/1.5094494) will be used to train the models
