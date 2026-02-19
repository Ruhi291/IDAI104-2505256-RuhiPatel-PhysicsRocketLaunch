# 🚀 Rocket Launch Path Visualization App  
Mathematics for AI – Summative Assessment  

## 📌 Project Overview

This project is a Streamlit-based web application designed to analyze and simulate rocket launch missions using real-world mission data and mathematical modeling.

The app combines:

- Data preprocessing and exploratory data analysis (EDA)
- Interactive mission visualizations
- A physics-based rocket launch simulation using Newton’s Second Law
- A clean and user-friendly dashboard interface

The goal of this project is to demonstrate how mathematical principles, particularly calculus and physics, can be applied to real-world aerospace systems and visualized using Python.

---

## 🧠 Physics & Mathematical Model

The rocket simulation in this application is based on:

### Newton’s Second Law:
\[
F = m \times a
\]

Rearranged as:

\[
a = \frac{F}{m}
\]

Where:

- **Thrust** acts upward
- **Gravity** acts downward
- **Drag** resists motion
- **Mass decreases** as fuel burns

Acceleration is calculated as:

\[
Acceleration = \frac{(Thrust - Gravity - Drag)}{Mass}
\]

The simulation updates:

- Acceleration
- Velocity
- Altitude
- Remaining mass

over multiple time steps to model a realistic rocket launch.

---

## 📊 Features of the App

### 🔎 Data Cleaning & Preprocessing
- Converts launch dates to proper datetime format
- Ensures numeric columns are correctly formatted
- Removes duplicates
- Handles missing values
- Uses caching for performance optimization

---

### 📈 Interactive Visualizations

The dashboard includes:

- **Scatter Plot:** Payload Weight vs Fuel Consumption  
- **Bar Chart:** Mission Cost (Success vs Failure)  
- **Line Chart:** Mission Duration vs Distance from Earth  
- **Box Plot:** Crew Size vs Mission Success  
- **Scatter Plot:** Scientific Yield vs Mission Cost  
- **Correlation Heatmap** of mission variables  

All graphs dynamically update based on user-selected filters.

---

### 🎛 Interactive Filters

Users can filter mission data using:

- Mission Type
- Launch Vehicle
- Distance Range
- Year Range

All charts update automatically based on selected filters.

---

### 🚀 Rocket Launch Simulation

The simulation section allows users to adjust:

- Thrust
- Initial Mass
- Fuel Mass
- Drag Coefficient

When the **“Launch Simulation”** button is pressed:

- The rocket trajectory is calculated
- Altitude vs Time graph is generated
- Mass reduction due to fuel burn is applied
- Results are visualized in real-time

The simulation only runs when triggered to ensure smooth performance.

---

### 🌗 Light & Dark Mode

The app includes a theme toggle allowing users to switch between:

- Light Mode
- Dark Mode

This improves usability and user experience.

---

## ⚙️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly

---

## 🌐 Deployment

The application is deployed using **Streamlit Cloud**.

🔗 Live App Link:  
(Add your Streamlit Cloud link here)

---

## 🎯 Conclusion

This project demonstrates the integration of mathematics, physics, and data science in solving real-world aerospace problems.

By combining:

- Statistical analysis
- Interactive visualizations
- A physics-based simulation model

the application provides meaningful insights into rocket mission dynamics and resource relationships.

The project reflects how mathematical modeling and AI-driven tools can be applied to engineering, aerospace analytics, and data-driven decision-making.

---

## 👩‍💻 Author

Student Name:  
Course: Artificial Intelligence  
Module: Mathematics for AI  
