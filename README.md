![alt text](images/app_logo.png)

## Table of Contents
1. [**Project Overview**](https://github.com/baclausen/final_project?tab=readme-ov-file#project-overview)
2. [**Original Data**](https://github.com/baclausen/final_project?tab=readme-ov-file#original-data)
3. [**Exploratory Analysis**](https://github.com/baclausen/final_project?tab=readme-ov-file#exploratory-analysis)
4. [**Machine Learning**](https://github.com/baclausen/final_project?tab=readme-ov-file#machine-learning)
5. [**Live Predictions**](https://github.com/baclausen/final_project?tab=readme-ov-file#live-predictions)
6. [**Automated Reporting**](https://github.com/baclausen/final_project?tab=readme-ov-file#automated-reporting)
7. [**Sources**](https://github.com/baclausen/final_project?tab=readme-ov-file#sources)

## Project Overview
Digital piracy in the United States is a persistent and costly issue, primarily impacting the film, TV, and music sectors. It costs the U.S. economy billions of dollars annually in lost revenue and contributes to substantial job losses, with illegal streaming now the dominant method of infringement. Despite the availability of numerous legal streaming platforms, high costs and content fragmentation are cited as reasons, particularly by younger consumers, who continue to drive unauthorized content consumption. The fight against piracy focuses on reinforcing intellectual property protections and coordinated global enforcement efforts.

This exploratory data analysis observes trends and/correlations in pirated video files to identify what categories of video content are the most susceptible.

## Original Data
- How has piracy grown since inception?
- What correlations exist between the amount of time a file is posted, the number of downloads, download rate, and IMDb rating?

## Exploratory Analysis
The original dataset contained a lot of poorly formatted data. Nearly all values, to include dates, counts of views/downloads, runtimes were stored as strings with inconsistent formats.

## Machine Learning
![alt text](images/top_ten_directors.png) <br>
A list of directors whose collective works have been downloaded the most.

![alt text](images/heatmap.png) <br>
A direct correlation with sheer number of downloads compared to views is apparent as a view must occur to initiate a download.
I was surprised the correlation between the download rate and IMDb rating was not stronger, as popular content would logically have a higher rate for piracy.
I had also anticipated a stronger relationship between the days between release/upload and the download rate to be higher. The concept of content available for piracy close to the release date being popular seems likely, but this could be skewed by data included before sites became mainstream.

![alt text](images/views_vs_downloads_scatter.png) <br>
Representation of the observation above showing a very strong logical connection between views and downloads.

![alt text](images/dl_rate_vs_days_to_upload.png) <br>
The information to glean from this graph lies heavily on the right side - we see that higher download percentages do, in fact, correspond to fast uploading in many cases. The cluster in the middle likely represents content created before the popularity of digital piracy.

![alt text](images/avg_upload_time_by_year.png) <br>
A line chart depicting the average amount of days between a release date and an upload date, categorized by year. The increase in the late 1990s coincides with the enactment of the Digital Millennium Copywrite Act, which outlined legal ramification of digital piracy. The chart flattens out near 2012, which is an *average* of near zero days between theatrical release and piracy and constitutes its dominance on the entertainment industry.

![alt text](images/release_vs_upload_hist.png) <br>
This overlay shows a surge in content available on the pirating site which indicates a huge uptick in adoption of illegal downloading. Aside from sheer volume, this is apparent when noting the number of videos posted surpassed US production, indicating that purveyors of pirated content were able to not onl keep pace with Hollywood, but access or digitize content from previous years.

## Live Predictions
The reason for the large surge around 2010 isn't evident within the dataframe, but further research aligns with several contributing factors: <br>

## Automated Reporting
The original data was sourced from Kaggle and uploaded by user Arsalan ur Rehman. <br>
The image in the header is AI generated from Google Gemini

## Sources <br>
