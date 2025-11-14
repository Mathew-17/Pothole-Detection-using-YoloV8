# 📌 Project Description

This project presents an AI-driven Pothole Detection and Heatmap Visualization System developed using YOLOv8, geospatial analytics, and cloud-based visualization tools. The solution automates traditional road inspection processes by detecting potholes in real-time from images or dashcam video streams.

A custom dataset of annotated road surface images is used to fine-tune the YOLOv8 model for reliable pothole detection under various lighting, weather, and road conditions. Each detection is enriched with GPS metadata and stored in a Supabase (PostgreSQL) cloud database for further analysis.

The system generates interactive heatmaps that show pothole density and potential risk areas. These visualizations, created using Grafana, help municipal authorities and road maintenance teams prioritize repairs, reduce manual surveying time, and improve overall transportation safety.

## 🔍 Key Features
- YOLOv8-based pothole detection from images and videos  
- Automated geo-tagging of potholes using GPS metadata  
- Secure cloud storage using Supabase PostgreSQL  
- Heatmap visualization of pothole density using Grafana  
- Real-time monitoring capability for smart-city applications  
- Scalable pipeline for continuous road condition assessment  

## 🧰 Tech Stack
- YOLOv8 (Ultralytics)
- Python
- OpenCV
- Supabase (PostgreSQL)
- Grafana
- GPS Integration


## 🎯 Impact
This system enables automated, data-driven decision-making in road maintenance by reducing manual inspection efforts, identifying high-risk road segments, and improving public safety through continuous monitoring.

