# Global Horizontal Irradiance (GHI) Nowcasting and Short-Term Forecasting Using the Folsom Dataset.
This project is my undergraduate thesis at Universidade Federal de Santa Catarina (UFSC) for my Electrical Engineering degree, and it is dedicated to my friends, family and the professors that guided me throughout the course.

---
## Motivation
As technology advances and our electrical devices become more and more powerful, the demand for electric energy increases. Our job, as engineers, is to supply this ever-increasing demand in a safe, sustainable and renewable way. It is a well known fact that solar photovoltaic energy is both sustainable and renewable, however, it's large-scale integration into the electric system still represents a safety concern to the grid operators around the world. This is mainly due to the highly unpredictable and fast-changing movement of the clouds that continuously block and unblock the sun from the solar farms, causing it's power output to be greatly irregular. If this irregularity is not accounted for by the grid operator, then there will be an inbalance between the supply and demand, which can cause the frequency/voltage of the system to exceed it's tolerated limits. Depending on how much these limits are exceeded by, the grid may suffer catastrophic mechanical and eletrical consequences. 

The aim of this project is to train deep learning models to nowcast and forecast the GHI 5 minutes ahead of time. To do this, the Folsom dataset (https://doi.org/10.1063/1.5094494) will be used to train the models. This dataset provides many different types of data, the main ones that will be used are the sky images (as inputs to the models) and the GHI measurements (as labels to the images). The reason for using the images as inputs to the models is because they provide information on the clouds with a high spatiotemporal resolution, which is necessary when forecasting on such short time horizons. With these predictions, the goal is twofold:

1. Provide these predictions to the grid operator so as to assist them in their real-time decisions and operation of the electric grid;

2. Be able to predict when a cloud is going to block/unblock the sun, making the highly irregular power output of a solar farm a predictable one.

By achieving both of these goals, solar photovoltaic energy will become much more predictable (perhaps even dispatchable if the predictions are extremely accurate) and it's large-scale integration into the electric grid will become much safer, therefore allowing it's greater penetration into the system. With this, the ever-increasing demand will be supplied by solar photovoltaic energy, and society will be able to develop in a sustainable way.

## Dataset
As mentioned in the previous section, the dataset used in this project is the Folsom, California dataset, created by [Pedro et al.][cc-by]

[cc-by]:  https://doi.org/10.1063/1.5094494

