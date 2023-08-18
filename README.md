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
As mentioned in the previous section, the dataset used in this project is the Folsom dataset, created by [Pedro et al.][cc-by]. This dataset provides three years (2014-2016) of 1-min resolution GHI measurements, as well as overlapping exogenous data such as: sky images; satellite images; Numerical Weather Prediction; weather data. Currently, the only exogenous data being used are the sky images. These sky images are in the Red-Green-Blue (RGB) color format and have a 1536x1536 pixel resolution. This resolution is downscaled to 64x64 in a preprocessing pipeline to save training time and memory consumption. The utilized data is summarized in the table below:

| File | Description |
| ------------- | ------------- |
| Folsom_irradiance.csv | 1-min resolution GHI, DNI and DHI data for the year 2014, 2015 and 2016. | 
| Folsom_sky_images_2014.tar.bz2 | Compressed 2014 sky images. After extraction each month will be a directory, and each day will be a sub-directory. The image files will be named acccording to the following format 2014MMDD_HHMMSS.jpg
| Folsom_sky_images_2015.tar.bz2 | Compressed 2015 sky images. After extraction each month will be a directory, and each day will be a sub-directory. The image files will be named acccording to the following format 2015MMDD_HHMMSS.jpg
| Folsom_sky_images_2016.tar.bz2 | Compressed 2016 sky images. After extraction each month will be a directory, and each day will be a sub-directory. The image files will be named acccording to the following format 2016MMDD_HHMMSS.jpg

[cc-by]:  https://doi.org/10.1063/1.5094494

## Dataset inconsistencies
The name of each image files sugests the exact time where the image was taken, so the image file `20161012_224059.jpg` is supposed to have been taken on `2016/10/12` at `22:40:59` (in UTC timezone). However, through some exploratory data analysis (EDA), the image files seem to be incorrectly named. This is verified by checking the date modified metadata of each image file and recognizing that they are indeed different, as can be seen in the example below.

<div align=center><image src="./repo_images/date_modified.png"></div>

It can be seen that, for the image file `20161012_224059.jpg`, there seems to be a 11 minute and 31 second difference between the image file name and it's date modified metadata. This inconsistency seems to be even more transparent when the images are overlapped with their corresponding GHI measurements. When using the file name to label the images, the GHI is high in some instances where the sun is covered but low on others where the sun is exposed. A comparison between the two is shown below.

![date modified gif](/repo_images/date_modif.gif)
<p align=center>
Figure 1: Images with date modified timestamps and corresponding GHI measurement
</p>

![img file gif](/repo_images/img_file.gif)
<p align=center>
Figure 2: Images with image file name timestamps and corresponding GHI measurement
</p>
