
# 🚇 Urban Train Route Optimization: Guadalajara (ZMG) 🇲🇽  
### A Bio-inspired Approach Using Machine Learning & Graph Theory

## 📋 Overview
This project explores an optimal route design for a new light-rail line in the Guadalajara Metropolitan Area (ZMG). Instead of relying purely on manual heuristics, it combines **Unsupervised Machine Learning**, **Geospatial Analysis**, and a **Bio-inspired Optimization Algorithm** to propose a data-driven, efficient, circular transit route.

The system extracts and processes geospatial data from **OpenStreetMap (OSM)**, identifies demand clusters based on Points of Interest (POIs), and computes an optimal route using a custom **Ant Colony Optimization (ACO)** implementation over the real street network.

---

## 🚀 Key Features

### **• Geospatial Analysis**
Collects POIs such as schools, hospitals, offices, and services across the five main municipalities:  
**Guadalajara, Zapopan, Tlaquepaque, Tonalá, and Tlajomulco**, using `osmnx`.

### **• Demand Clustering**
Uses **K-Means** to determine strategic station centroids based on POI density.

### **• Real Network Distance Calculation**
Applies **Dijkstra’s Algorithm** via `NetworkX` to compute shortest-path street distances (avoiding Euclidean inaccuracies).

### **• Bio-inspired Route Optimization**
A custom **ACO** class solves the TSP-like problem of finding the most efficient route visiting all stations.

### **• Visualization**
Generates:
- Density heatmaps (`seaborn`)
- Final optimized route overlaid on the city map (`matplotlib` + `osmnx`)

---

## 🛠️ Tech Stack

**Language:** Python  
**Geospatial:** OSMnx, GeoPandas, Shapely  
**Machine Learning:** Scikit-learn (K-Means)  
**Graph Theory:** NetworkX  
**Visualization:** Matplotlib, Seaborn  

---

## ⚙️ Methodology Pipeline

### **1. Data Ingestion**
- Download the street network and POIs for target municipalities from OSM.

### **2. Preprocessing**
- Convert POI geometries into usable coordinate points.
- Filter and standardize datasets.

### **3. Station Optimization**
- Apply **K-Means (k = 20)** to identify station centroids from high-density demand zones.

### **4. Cost Matrix Generation**
- Compute shortest-path distances between every pair of station centroids using the OSM-based network graph.

### **5. ACO Route Optimization**
- Initialize ant agents to explore possible paths.  
- Update pheromones proportional to path quality.  
- Apply evaporation to converge on an optimal circular route.

### **6. Visualization**
- Render the optimized route across the city map using real coordinates and the street graph.

---

## 📊 Visuals & Results


- **Figure 1:** Density Heatmap & Proposed Stations (Cyan X)
 <img width="1729" height="993" alt="image" src="https://github.com/user-attachments/assets/957f6b19-630f-494c-8dc4-806b3fe2be49" />

- **Figure 2:** Final Optimized Route (ACO Result)
 <img width="1312" height="1587" alt="image" src="https://github.com/user-attachments/assets/f7d0d91f-4c2c-4f6d-9e67-dbd0bdd192ca" />


---

## 📦 Installation & Usage

### **1. Clone the repository**
```bash
git clone https://github.com/NullPointerException-25/Urban-Train-Route-Optimization.git
````

### **2. Install dependencies**

```bash
pip install osmnx pandas numpy matplotlib scikit-learn geopandas seaborn networkx
```

### **3. Run the script**

```bash
python main.py
```

> **Note:** Initial OSM downloads can take several minutes depending on internet speed.

---

## 🔮 Future Improvements

* Integration with real-time traffic APIs (e.g., Google Maps, Waze) to dynamically adjust edge weights.
* Multi-objective optimization (distance vs. construction cost).
* Interactive dashboard or web-based UI using **Streamlit**.
* Incorporating socioeconomic layers for improved demand modeling.

---

## 👤 Author

**Samuel Isaí Ramos Díaz**

* [LinkedIn](www.linkedin.com/in/samdiaz5656)

```
```
