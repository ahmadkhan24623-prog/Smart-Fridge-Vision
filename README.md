# 🌿 Smart Fridge Vision & Culinary AI Engine
**Pir Mehr Ali Shah Arid Agriculture University Rawalpindi (PMAS-AAUR)**  
*Department of Computer Science & Information Technology*  
*Course: Artificial Intelligence (CSC-203) — 3rd Semester Project*

---

## 📖 Overview
The **Smart Fridge Vision & Culinary AI Engine** is an intelligent end-to-end culinary system designed to eliminate domestic food waste and assist home cooks. By combining computer vision with semantic ingredient recommendation algorithms, the application allows users to photograph the contents of their fridge or pantry, automatically identifies ingredients, and computes tailored recipes with exact step-by-step cooking timers.

---

## 🚀 Key Features

1. **Neural Vision Ingredient Detection**:
   - Built on **PyTorch** and **MobileNetV3 Small** pre-trained on ImageNet.
   - Enhanced with color-analytics heuristics to identify fresh cuts (beef, chicken, seafood), vegetables, and breakfast items.
   - Real-time webcam scanning or high-resolution photo drag-and-drop.

2. **Smart Fridge & Pantry Inventory**:
   - Visual inventory tracking with detected item tags and confidence scores.
   - 1-Click Quick Add buttons for instant manual adjustments.
   - Support for 22 core kitchen pantry classes.

3. **Protein-Integrity Jaccard Semantic Matcher**:
   - Advanced heuristic matching algorithm that prevents invalid recipe recommendations (e.g., won't suggest beef steak if only poultry is detected).
   - Dynamic match percentage scoring with visual badge indicators.
   - Categorical filtering (Non-Veg, Seafood, Breakfast, Fast & Easy).

4. **Chef-Grade Interactive Cooking Studio**:
   - Ingredient checklist with exact measurements and pantry availability banners.
   - 4-Stage step-by-step interactive cooking guides with dedicated countdown timers.
   - Professional culinary chef tips for each dish.

---

## 🏗️ Project Architecture

```
Fridge-recipe-generator/
│
├── app.py                  # Main Streamlit web application & high-end UI
├── model_loader.py         # PyTorch MobileNetV3 deep learning vision engine
├── recommender.py          # Jaccard semantic matching & recommendation logic
├── recipes.csv             # Curated dataset of culinary recipes and instructions
├── requirements.txt        # Python dependency manifest
├── .streamlit/
│   └── config.toml         # Custom Streamlit server & theme configuration
└── samples/                # Sample test images for offline/instant demonstrations
    ├── raw_chicken.png
    ├── raw_beef_steak.png
    ├── pan_fried_fish.png
    ├── chicken_curry.jpg
    ├── beef_steak.jpg
    ├── breakfast_eggs.jpg
    ├── fried_fish.jpg
    └── garlic_rice.jpg
```

---

## 💻 Installation & Local Setup

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Activate Virtual Environment
```powershell
# In PowerShell (Windows):
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```
The application will automatically open in your default browser at:
`http://localhost:8501`

---

## 🔬 Technologies Used
- **Deep Learning Framework**: PyTorch (`torch`, `torchvision`)
- **Neural Backbone**: MobileNetV3 Small
- **Web UI & State Management**: Streamlit
- **Data Engineering**: Pandas, NumPy
- **Image Processing**: Pillow (PIL)

---

## 👥 Authors & Academic Credits
* **Institution**: Pir Mehr Ali Shah Arid Agriculture University Rawalpindi
* **Department**: Computer Science & Information Technology
* **Course**: CSC-203 (Artificial Intelligence)
